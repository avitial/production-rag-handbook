"""End-to-end tests for validated and serialized RAG answers."""

from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image

from app.api.schemas import (
    CitationSchema,
    RAGAnswerResponse,
    ValidationIssueSchema,
    ValidationSummarySchema,
)
from app.embeddings.sentence_transformer import (
    DeterministicHashEmbeddingProvider,
)
from app.generation.context_builder import ContextBuilder
from app.generation.local_llm_client import (
    DeterministicLocalLLMClient,
)
from app.generation.rag_generator import RAGGenerator
from app.ingestion.pipeline import (
    IngestionConfig,
    LocalIngestionPipeline,
)
from app.reranking.base import RerankingRequest
from app.reranking.cross_encoder import (
    DeterministicReranker,
)
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import (
    SearchFilters,
    VectorSearchRequest,
)
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry
from app.validation.answer_validator import validate_answer
from app.validation.json_validator import (
    validate_rag_response_json,
)


HANDWRITTEN = Path("/mnt/data/SYN-200849_handwritten.png")
NATIVE_PDF = Path("/mnt/data/SYN-200989.pdf")


def fake_ocr(_image: Image.Image, *, language: str) -> str:
    assert language == "eng"
    return (
        "*** SYNTHETIC MEDICAL RECORD ***\n"
        "Patient Demographics\n"
        "Patient ID: SYN-200849\n"
        "Clinical Notes (SOAP)\n"
        "A: Routine exam\n"
        "P: Follow-up appointment in 6 months.\n"
        "Medications\n"
        "Current: Metformin\n"
        "Allergies\n"
        "Latex\n"
        "Problems\n"
        "Active Diagnosis: Routine exam\n"
    )


def make_samples(tmp_path: Path) -> Path:
    samples = tmp_path / "samples"
    samples.mkdir()
    shutil.copy2(
        HANDWRITTEN,
        samples / "SYN-200849_handwritten.png",
    )
    shutil.copy2(
        NATIVE_PDF,
        samples / "SYN-200989.pdf",
    )
    return samples


def test_validated_allergy_response_round_trips(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    provider = DeterministicHashEmbeddingProvider()
    store = ChromaStore(
        embedding_provider=provider,
        collection_name="day10",
        persistence_directory=tmp_path / "vectors",
        backend="local",
    )
    ingestion = LocalIngestionPipeline(
        chroma_store=store,
        registry=DocumentRegistry(
            tmp_path / "registry.sqlite3"
        ),
        config=IngestionConfig(
            max_characters=300,
            overlap_characters=40,
        ),
        ocr_function=fake_ocr,
    )
    report = ingestion.ingest(samples)
    assert report.failed_files == 0

    bm25 = BM25Retriever(store)
    bm25.rebuild()
    hybrid = HybridRetriever(
        vector_retriever=VectorRetriever(store),
        bm25_retriever=bm25,
    )
    question = "What allergies are documented?"
    retrieved = hybrid.search(
        VectorSearchRequest(
            query=question,
            top_k=10,
            filters=SearchFilters(
                patient_id="SYN-200849"
            ),
        )
    )
    reranked = DeterministicReranker().rerank(
        RerankingRequest(
            query=question,
            passages=retrieved.results,
            top_n=5,
        )
    )
    answer = RAGGenerator(
        llm_client=DeterministicLocalLLMClient(),
        context_builder=ContextBuilder(),
    ).generate(
        question=question,
        passages=reranked.results,
    )

    answer_validation = validate_answer(
        answer,
        expected_patient_id="SYN-200849",
    )
    assert answer_validation.valid is True

    response = RAGAnswerResponse(
        question=answer.question,
        answer=answer.answer,
        citations=tuple(
            CitationSchema(
                source_number=item.source_number,
                chunk_id=item.chunk_id,
                filename=item.filename,
                page_number=item.page_number,
                section=item.section,
                patient_id=item.patient_id,
                citation_label=item.citation_label,
            )
            for item in answer.citations
        ),
        abstained=answer.diagnostics.abstained,
        validation=ValidationSummarySchema(
            valid=answer_validation.valid,
            citation_valid=(
                answer_validation.citation_result.valid
            ),
            answer_grounded=answer_validation.grounded,
            json_valid=True,
            issues=tuple(
                ValidationIssueSchema(
                    code=issue.code,
                    message=issue.message,
                    severity=issue.severity,
                    details=issue.details,
                )
                for issue in answer_validation.issues
            ),
        ),
        diagnostics={
            "llm_model": answer.diagnostics.llm_model,
            "context_sources": (
                answer.diagnostics.context_source_count
            ),
        },
    )

    json_result = validate_rag_response_json(response)

    assert json_result.valid is True
    assert json_result.normalized["answer"]
    assert json_result.normalized["citations"][0][
        "patient_id"
    ] == "SYN-200849"
