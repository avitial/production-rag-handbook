"""Confidence-gated question-answering endpoint."""

from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import RuntimeServices, get_services
from app.api.schemas import (
    CitationApi,
    ConfidenceReasonApi,
    QuestionApiRequest,
    QuestionApiResponse,
)
from app.confidence.features import extract_confidence_features
from app.generation.ollama_llm_client import OllamaBackendError
from app.reranking.base import RerankingRequest
from app.retrieval.models import SearchFilters, VectorSearchRequest
from app.validation.answer_validator import validate_answer
from app.validation.json_validator import validate_rag_response_json


router = APIRouter(prefix="/questions", tags=["questions"])


def _filters(values) -> SearchFilters:
    return SearchFilters(
        patient_id=values.patient_id,
        document_id=values.document_id,
        filename=values.filename,
        source_format=values.source_format,
        page_number=values.page_number,
    )


@router.post("", response_model=QuestionApiResponse)
def ask_question(
    request: QuestionApiRequest,
    services: RuntimeServices = Depends(get_services),
) -> QuestionApiResponse:
    total_started = perf_counter()

    retrieval_started = perf_counter()
    retrieval = services.hybrid_retriever.search(
        VectorSearchRequest(
            query=request.question,
            top_k=request.candidate_k,
            filters=_filters(request.filters),
        )
    )
    retrieval_ms = (perf_counter() - retrieval_started) * 1000

    reranking = services.reranker.rerank(
        RerankingRequest(
            query=request.question,
            passages=retrieval.results,
            top_n=request.final_k,
        )
    )
    try:
        answer = services.generator.generate(
            question=request.question,
            passages=reranking.results,
        )
    except OllamaBackendError as exc:
        services.logger.log(
            "ollama_generation_failed",
            fields={"error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    answer_validation = validate_answer(
        answer,
        expected_patient_id=request.filters.patient_id,
    )

    json_validation = validate_rag_response_json(
        {
            "question": answer.question,
            "answer": answer.answer,
            "citations": [
                {
                    "source_number": item.source_number,
                    "filename": item.filename,
                    "page_number": item.page_number,
                }
                for item in answer.citations
            ],
            "abstained": answer.diagnostics.abstained,
            "validation": {"valid": answer_validation.valid},
            "diagnostics": {},
        }
    )

    features = extract_confidence_features(
        retrieval_response=retrieval,
        reranking_response=reranking,
        rag_answer=answer,
        answer_validation=answer_validation,
        json_validation=json_validation,
    )
    assessment = services.confidence_policy.evaluate(features)

    total_ms = (perf_counter() - total_started) * 1000
    services.metrics.increment("question_requests_total")
    services.metrics.increment(
        f"question_decision_{assessment.decision.value}_total"
    )
    services.metrics.observe_latency("api_question_total_ms", total_ms)
    services.logger.log(
        "question_completed",
        fields={
            "decision": assessment.decision.value,
            "reason_codes": [
                reason.code for reason in assessment.reasons
            ],
            "patient_filter_applied": bool(request.filters.patient_id),
            "retrieval_candidates": len(retrieval.results),
            "final_context_sources": len(answer.context.sources),
            "total_ms": total_ms,
        },
    )

    return QuestionApiResponse(
        question=request.question,
        answer=answer.answer,
        decision=assessment.decision.value,
        abstained=answer.diagnostics.abstained,
        citations=[
            CitationApi(
                source_number=item.source_number,
                chunk_id=item.chunk_id,
                filename=item.filename,
                page_number=item.page_number,
                section=item.section,
                patient_id=item.patient_id,
                citation_label=item.citation_label,
            )
            for item in answer.citations
        ],
        reasons=[
            ConfidenceReasonApi(
                code=reason.code,
                message=reason.message,
                details=reason.details,
            )
            for reason in assessment.reasons
        ],
        validation={
            "answer_valid": answer_validation.valid,
            "citation_valid": answer_validation.citation_result.valid,
            "answer_grounded": answer_validation.grounded,
            "json_valid": json_validation.valid,
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity,
                    "details": issue.details,
                }
                for issue in answer_validation.issues
            ],
        },
        confidence_features=features.to_dict(),
        diagnostics={
            "retrieval_ms": retrieval_ms,
            "reranking_ms": reranking.diagnostics.duration_ms,
            "generation_ms": answer.diagnostics.duration_ms,
            "total_ms": total_ms,
            "retrieval_model": retrieval.diagnostics.embedding_model,
            "reranker_model": reranking.diagnostics.model_name,
            "generator_model": answer.diagnostics.llm_model,
        },
    )
