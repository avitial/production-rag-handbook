"""Unit tests for citation-aware context construction."""

from app.generation.context_builder import (
    ContextBuilder,
    ContextBuilderConfig,
)
from app.reranking.base import RerankedPassage
from app.retrieval.models import RetrievedPassage


def make_passage(
    chunk_id: str,
    *,
    text: str,
    rank: int = 1,
    patient_id: str = "SYN-001",
    filename: str = "sample.pdf",
    section: str | None = "Allergies",
) -> RerankedPassage:
    retrieved = RetrievedPassage(
        chunk_id=chunk_id,
        document_id="doc-001",
        filename=filename,
        source_path=f"/tmp/{filename}",
        source_format="pdf",
        page_number=1,
        section=section,
        patient_id=patient_id,
        text=text,
        rank=rank,
        distance=0.2,
        similarity=0.8,
    )
    return RerankedPassage(
        passage=retrieved,
        rerank_rank=rank,
        rerank_score=1.0 / rank,
        original_rank=rank,
        original_similarity=0.8,
        model_name="test-reranker",
    )


def test_builds_numbered_sources_with_provenance() -> None:
    context = ContextBuilder().build(
        [
            make_passage(
                "c1",
                text="Allergies\nLatex",
            ),
            make_passage(
                "c2",
                text="Medications\nMetformin",
                rank=2,
                section="Medications",
            ),
        ]
    )

    assert "[SOURCE 1]" in context.text
    assert "File: sample.pdf" in context.text
    assert "Page: 1" in context.text
    assert "Patient ID: SYN-001" in context.text
    assert len(context.sources) == 2
    assert context.citation_map[1].chunk_id == "c1"


def test_duplicate_chunks_are_included_once() -> None:
    passage = make_passage(
        "duplicate",
        text="Allergies\nLatex",
    )

    context = ContextBuilder().build(
        [passage, passage]
    )

    assert len(context.sources) == 1
    assert context.text.count("[SOURCE 1]") == 1


def test_source_limit_is_enforced() -> None:
    builder = ContextBuilder(
        ContextBuilderConfig(
            maximum_sources=2,
            maximum_characters=5000,
        )
    )

    context = builder.build(
        [
            make_passage("c1", text="one"),
            make_passage("c2", text="two", rank=2),
            make_passage("c3", text="three", rank=3),
        ]
    )

    assert len(context.sources) == 2
    assert context.omitted_source_count == 1


def test_character_budget_stops_before_partial_source() -> None:
    builder = ContextBuilder(
        ContextBuilderConfig(
            maximum_characters=180,
            maximum_sources=5,
            allow_truncation=False,
        )
    )

    context = builder.build(
        [
            make_passage(
                "c1",
                text="Allergies\nLatex",
            ),
            make_passage(
                "c2",
                text="X" * 400,
                rank=2,
            ),
        ]
    )

    assert len(context.sources) == 1
    assert context.truncated is False
    assert "X" * 100 not in context.text


def test_optional_truncation_marks_context() -> None:
    builder = ContextBuilder(
        ContextBuilderConfig(
            maximum_characters=180,
            maximum_sources=2,
            allow_truncation=True,
        )
    )

    context = builder.build(
        [
            make_passage(
                "c1",
                text="Long passage " * 50,
            )
        ]
    )

    assert len(context.sources) == 1
    assert context.truncated is True
    assert context.included_characters <= 180