"""End-to-end integration tests for vector + BM25 + RRF retrieval."""

from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.embeddings.sentence_transformer import (
    DeterministicHashEmbeddingProvider,
)
from app.ingestion.pipeline import (
    IngestionConfig,
    LocalIngestionPipeline,
)
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import (
    HybridRetrievalConfig,
    HybridRetriever,
)
from app.retrieval.models import (
    SearchFilters,
    VectorSearchRequest,
)
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry


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
        collection_name="day7_documents",
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
        config=HybridRetrievalConfig(
            vector_top_k=10,
            keyword_top_k=10,
            fusion_constant=60,
        ),
    )
    return ingestion, store, vector, bm25, hybrid


def test_bm25_rebuilds_from_ingested_chunks(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, store, _vector, bm25, _hybrid = make_system(
        tmp_path
    )
    report = ingestion.ingest(samples)

    assert report.failed_files == 0
    assert bm25.rebuild() == store.count()
    assert bm25.document_count == store.count()


def test_bm25_finds_exact_patient_identifier(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, _store, _vector, bm25, _hybrid = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    bm25.rebuild()

    response = bm25.search(
        VectorSearchRequest(
            query="SYN-200849",
            top_k=5,
        )
    )

    assert response.results
    assert response.results[0].patient_id == "SYN-200849"
    assert "SYN-200849" in response.results[0].text


def test_hybrid_retrieval_returns_unique_ranked_chunks(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, _store, _vector, _bm25, hybrid = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    response = hybrid.search(
        VectorSearchRequest(
            query="What allergies are documented?",
            top_k=5,
        )
    )

    ids = [item.chunk_id for item in response.results]

    assert response.results
    assert len(ids) == len(set(ids))
    assert [item.rank for item in response.results] == list(
        range(1, len(response.results) + 1)
    )
    assert all(
        item.metadata["retrieval_method"] == "hybrid"
        for item in response.results
    )


def test_hybrid_patient_filter_prevents_cross_patient_results(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, _store, _vector, _bm25, hybrid = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    response = hybrid.search(
        VectorSearchRequest(
            query="What allergies are documented?",
            top_k=10,
            filters=SearchFilters(
                patient_id="SYN-200849"
            ),
        )
    )

    assert response.results
    assert all(
        item.patient_id == "SYN-200849"
        for item in response.results
    )
    assert all(
        "Shellfish" not in item.text
        for item in response.results
    )


def test_keyword_and_hybrid_find_exact_allergy(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, _store, _vector, bm25, hybrid = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    request = VectorSearchRequest(
        query="Latex allergy",
        top_k=3,
        filters=SearchFilters(
            patient_id="SYN-200849"
        ),
    )
    keyword = bm25.search(request)
    fused = hybrid.search(request)

    assert keyword.results
    assert "Latex" in keyword.results[0].text
    assert fused.results
    assert any(
        "Latex" in result.text
        for result in fused.results
    )


def test_unanswerable_cardiology_query_does_not_create_false_metadata(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    ingestion, _store, _vector, _bm25, hybrid = make_system(
        tmp_path
    )
    ingestion.ingest(samples)
    hybrid.rebuild_keyword_index()

    response = hybrid.search(
        VectorSearchRequest(
            query="Which patient had a cardiology referral?",
            top_k=5,
        )
    )

    # Retrieval can return related passages, but no result may invent a
    # cardiology referral or a field that does not exist in source metadata.
    assert all(
        "cardiology referral" not in item.text.lower()
        for item in response.results
    )
