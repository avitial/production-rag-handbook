"""End-to-end tests for retrieval, reranking, and grounded generation."""

from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image

from app.embeddings.sentence_transformer import (
    DeterministicHashEmbeddingProvider,
)
from app.generation.context_builder import (
    ContextBuilder,
    ContextBuilderConfig,
)
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


HANDWRITTEN = Path("/home/avitial/workspace/RAG/production-rag-handbook/data/samples/handwritten/SYN-200849_handwritten.png")
NATIVE_PDF = Path("/home/avitial/workspace/RAG/production-rag-handbook/data/samples/native-pdf/SYN-200989.pdf")


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

    if HANDWRITTEN.exists():
        shutil.copy2(
            HANDWRITTEN,
            samples / "SYN-200849_handwritten.png",
        )
    else:
        Image.new("RGB", (200, 200), "white").save(
            samples / "SYN-200849_handwritten.png"
        )

    if NATIVE_PDF.exists():
        shutil.copy2(
            NATIVE_PDF,
            samples / "SYN-200989.pdf",
        )

    return samples


def make_system(tmp_path: Path):
    provider = DeterministicHashEmbeddingProvider(
        dimensions=128
    )
    store = ChromaStore(
        embedding_provider=provider,
        collection_name="day9_documents",
        persistence_directory=tmp_path / "vectors",
        backend="local",
    )
    registry = DocumentRegistry(
        tmp_path / "registry.sqlite3"
    )
    ingestion = LocalIngestionPipeline(
        chroma_store=store,
        registry=registry,
        config=IngestionConfig(
            max_characters=300,
            overlap_characters=40,
        ),
        ocr_function=fake_ocr,
    )
    vector = VectorRetriever(store)
    bm25 = BM25Retriever(store)
    hybrid = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
    )
    reranker = DeterministicReranker()
    generator = RAGGenerator(
        llm_client=DeterministicLocalLLMClient(),
        context_builder=ContextBuilder(
            ContextBuilderConfig(
                maximum_characters=3000,
                maximum_sources=4,
            )
        ),
    )
    return ingestion, hybrid, reranker, generator


def answer_question(
    *,
    hybrid,
    reranker,
    generator,
    question: str,
    patient_id: str | None,
):
    retrieved = hybrid.search(
        VectorSearchRequest(
            query=question,
            top_k=10,
            filters=SearchFilters(
                patient_id=patient_id
            ),
        )
    )
    reranked = reranker.rerank(
        RerankingRequest(
            query=question,
            passages=retrieved.results,
            top_n=5,
        )
    )
    return generator.generate(
        question=question,
        passages=reranked.results,
    )


def test_generates_allergy_answer_with_valid_citation(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, hybrid, reranker, generator = make_system(
        tmp_path
    )
    report = ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    assert report.failed_files == 0

    result = answer_question(
        hybrid=hybrid,
        reranker=reranker,
        generator=generator,
        question="What allergies are documented?",
        patient_id="SYN-200849",
    )

    assert "Latex" in result.answer
    assert "[SOURCE" in result.answer
    assert result.citations
    assert result.citations[0].patient_id == "SYN-200849"
    assert result.diagnostics.invalid_citation_numbers == ()
    assert result.diagnostics.abstained is False


def test_generates_follow_up_answer_from_handwritten_source(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, hybrid, reranker, generator = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    result = answer_question(
        hybrid=hybrid,
        reranker=reranker,
        generator=generator,
        question="When is the follow-up appointment?",
        patient_id="SYN-200849",
    )

    assert "6 months" in result.answer
    assert any(
        citation.filename == "SYN-200849_handwritten.png"
        for citation in result.citations
    )


def test_unsupported_cardiology_question_abstains(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, hybrid, reranker, generator = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    result = answer_question(
        hybrid=hybrid,
        reranker=reranker,
        generator=generator,
        question="Which patient had a cardiology referral?",
        patient_id=None,
    )

    assert "could not find enough explicit evidence" in (
        result.answer.lower()
    )
    assert result.diagnostics.abstained is True


def test_patient_filter_prevents_cross_patient_answer(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, hybrid, reranker, generator = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    result = answer_question(
        hybrid=hybrid,
        reranker=reranker,
        generator=generator,
        question="What allergies are documented?",
        patient_id="SYN-200849",
    )

    assert "Shellfish" not in result.answer
    assert all(
        citation.patient_id == "SYN-200849"
        for citation in result.citations
    )


def test_context_budget_and_source_limit_are_reported(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, hybrid, reranker, _generator = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    limited_generator = RAGGenerator(
        llm_client=DeterministicLocalLLMClient(),
        context_builder=ContextBuilder(
            ContextBuilderConfig(
                maximum_characters=450,
                maximum_sources=2,
            )
        ),
    )
    result = answer_question(
        hybrid=hybrid,
        reranker=reranker,
        generator=limited_generator,
        question="What medication is documented?",
        patient_id="SYN-200849",
    )

    assert result.diagnostics.context_source_count <= 2
    assert result.diagnostics.context_characters <= 450


def test_discharge_qualified_medication_question_abstains(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, hybrid, reranker, generator = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    result = answer_question(
        hybrid=hybrid,
        reranker=reranker,
        generator=generator,
        question="What medication was prescribed at discharge?",
        patient_id=None,
    )

    assert "could not find enough explicit evidence" in (
        result.answer.lower()
    )
    assert result.diagnostics.abstained is True
