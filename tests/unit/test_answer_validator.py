"""Unit tests for answer grounding validation."""

from app.generation.context_builder import (
    BuiltContext,
    ContextSource,
)
from app.generation.llm_client import LLMResponse
from app.generation.prompts import PromptBundle
from app.generation.rag_generator import (
    AnswerCitation,
    RAGAnswer,
    RAGGenerationDiagnostics,
)
from app.validation.answer_validator import validate_answer


def make_answer(
    *,
    answer_text: str,
    patient_id: str = "SYN-001",
    cited: bool = True,
    abstained: bool = False,
) -> RAGAnswer:
    source = ContextSource(
        source_number=1,
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="record.pdf",
        page_number=1,
        section="Allergies",
        patient_id=patient_id,
        text="Allergies\nLatex",
        citation_label="record.pdf, page 1, section Allergies",
        rerank_score=0.9,
    )
    context = BuiltContext(
        text="[SOURCE 1]\nAllergies\nLatex",
        sources=(source,),
        included_characters=30,
        omitted_source_count=0,
        truncated=False,
    )
    citations = (
        (
            AnswerCitation(
                source_number=1,
                chunk_id="chunk-1",
                filename="record.pdf",
                page_number=1,
                section="Allergies",
                patient_id=patient_id,
                citation_label=(
                    "record.pdf, page 1, section Allergies"
                ),
            ),
        )
        if cited
        else ()
    )

    return RAGAnswer(
        question="What allergies are documented?",
        answer=answer_text,
        citations=citations,
        context=context,
        prompts=PromptBundle(
            system_prompt="Use evidence.",
            user_prompt="Question and context.",
        ),
        llm_response=LLMResponse(
            text=answer_text,
            model_name="test-model",
            duration_ms=1.0,
            input_characters=10,
            output_characters=len(answer_text),
        ),
        diagnostics=RAGGenerationDiagnostics(
            llm_model="test-model",
            context_source_count=1,
            context_characters=30,
            omitted_source_count=0,
            context_truncated=False,
            citation_count=len(citations),
            invalid_citation_numbers=(),
            abstained=abstained,
            duration_ms=1.0,
        ),
    )


def test_grounded_answer_passes() -> None:
    result = validate_answer(
        make_answer(
            answer_text="Latex is documented. [SOURCE 1]"
        ),
        expected_patient_id="SYN-001",
    )

    assert result.valid is True
    assert result.grounded is True


def test_missing_citation_fails() -> None:
    result = validate_answer(
        make_answer(
            answer_text="Latex is documented.",
            cited=False,
        )
    )

    assert result.valid is False
    assert any(
        issue.code == "missing_citation"
        for issue in result.issues
    )


def test_cross_patient_citation_fails() -> None:
    result = validate_answer(
        make_answer(
            answer_text="Latex is documented. [SOURCE 1]",
            patient_id="SYN-OTHER",
        ),
        expected_patient_id="SYN-001",
    )

    assert result.valid is False
    assert any(
        issue.code == "cross_patient_citation"
        for issue in result.issues
    )


def test_unsupported_answer_without_overlap_fails() -> None:
    result = validate_answer(
        make_answer(
            answer_text=(
                "A cardiology referral was completed. [SOURCE 1]"
            )
        )
    )

    assert result.valid is False
    assert any(
        issue.code == "insufficient_evidence_overlap"
        for issue in result.issues
    )


def test_abstention_passes_without_citation() -> None:
    result = validate_answer(
        make_answer(
            answer_text=(
                "I could not find enough explicit evidence in the "
                "provided sources to answer this question."
            ),
            cited=False,
            abstained=True,
        )
    )

    assert result.valid is True
    assert result.abstained is True
    assert result.grounded is True
