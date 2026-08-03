"""End-to-end tests for the synchronized Day 5 package."""

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
        "A: Routine exam (Z00.00)\n"
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


def make_pipeline(
    tmp_path: Path,
    *,
    chunk_size: int = 300,
) -> tuple[LocalIngestionPipeline, ChromaStore, DocumentRegistry]:
    provider = DeterministicHashEmbeddingProvider(
        dimensions=128
    )
    store = ChromaStore(
        embedding_provider=provider,
        collection_name="test_documents",
        persistence_directory=tmp_path / "vectors",
        backend="local",
    )
    registry = DocumentRegistry(
        tmp_path / "registry.sqlite3"
    )
    pipeline = LocalIngestionPipeline(
        chroma_store=store,
        registry=registry,
        config=IngestionConfig(
            max_characters=chunk_size,
            overlap_characters=40,
        ),
        ocr_function=fake_ocr,
    )
    return pipeline, store, registry


def test_end_to_end_ingestion(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    pipeline, store, registry = make_pipeline(tmp_path)

    report = pipeline.ingest(samples)

    expected_files = 2 if NATIVE_PDF.exists() else 1
    assert report.discovered_files == expected_files
    assert report.processed_files == expected_files
    assert report.failed_files == 0
    assert report.page_count == expected_files
    assert report.chunk_count > 0
    assert report.indexed_chunk_count == store.count()
    assert registry.count() == expected_files


def test_metadata_and_documents_are_persisted(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    pipeline, store, _ = make_pipeline(tmp_path)
    pipeline.ingest(samples)

    stored = store.collection.get(
        include=["documents", "metadatas"]
    )
    documents = stored["documents"]
    metadatas = stored["metadatas"]

    assert any("Metformin" in item for item in documents)
    assert any(
        item.get("patient_id") == "SYN-200849"
        for item in metadatas
    )
    assert all("embedding_model" in item for item in metadatas)
    assert all("page_number" in item for item in metadatas)


def test_second_run_is_skipped(tmp_path: Path) -> None:
    samples = make_samples(tmp_path)
    pipeline, store, _ = make_pipeline(tmp_path)

    first = pipeline.ingest(samples)
    first_count = store.count()
    second = pipeline.ingest(samples)

    assert first.processed_files > 0
    assert second.processed_files == 0
    assert second.skipped_files == first.discovered_files
    assert store.count() == first_count


def test_changed_chunking_configuration_rebuilds(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    first_pipeline, store, registry = make_pipeline(
        tmp_path,
        chunk_size=300,
    )
    first = first_pipeline.ingest(samples)

    second_pipeline = LocalIngestionPipeline(
        chroma_store=store,
        registry=registry,
        config=IngestionConfig(
            max_characters=150,
            overlap_characters=20,
        ),
        ocr_function=fake_ocr,
    )
    second = second_pipeline.ingest(samples)

    assert first.processed_files > 0
    assert second.processed_files == first.discovered_files
    assert second.skipped_files == 0
    assert second.failed_files == 0


def test_all_sample_questions_return_ranked_results(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    pipeline, store, _ = make_pipeline(tmp_path)
    pipeline.ingest(samples)

    questions = [
        "What medication was prescribed at discharge?",
        "What allergies are documented?",
        "When is the follow-up appointment?",
        "What was the primary diagnosis?",
        "Which patient had a cardiology referral?",
    ]

    for question in questions:
        result = store.query(question, top_k=3)
        assert len(result["documents"][0]) >= 1
        assert len(result["metadatas"][0]) >= 1
        assert len(result["distances"][0]) >= 1


def test_persistence_survives_new_store_instance(
    tmp_path: Path,
) -> None:
    samples = make_samples(tmp_path)
    pipeline, first_store, _ = make_pipeline(tmp_path)
    pipeline.ingest(samples)
    count = first_store.count()

    provider = DeterministicHashEmbeddingProvider(
        dimensions=128
    )
    reopened = ChromaStore(
        embedding_provider=provider,
        collection_name="test_documents",
        persistence_directory=tmp_path / "vectors",
        backend="local",
    )

    assert reopened.count() == count
    assert reopened.query(
        "What allergies are documented?",
        top_k=2,
    )["documents"][0]
