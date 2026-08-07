"""End-to-end RAG generation using the Ollama adapter with a fake server.

This test exercises:
    PDF and handwritten-image ingestion
    OCR injection
    vector and BM25 indexing
    hybrid retrieval
    reranking
    numbered context creation
    Ollama chat request construction
    citation resolution and validation
"""

from pathlib import Path
import shutil

from PIL import Image

from app.api.dependencies import ApiSettings, RuntimeServices
from app.generation.ollama_llm_client import (
    OllamaLLMClient,
    OllamaLLMConfig,
)
from app.reranking.base import RerankingRequest
from app.retrieval.models import (
    SearchFilters,
    VectorSearchRequest,
)
from app.validation.answer_validator import validate_answer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDWRITTEN = (
    PROJECT_ROOT
    / "data/samples/handwritten/SYN-200849_handwritten.png"
)
NATIVE_PDF = (
    PROJECT_ROOT
    / "data/samples/native-pdf/SYN-200989.pdf"
)


class FakeOllamaServer:
    def __init__(self) -> None:
        self.last_request = None

    def list(self):
        return {"models": [{"model": "gemma3:4b"}]}

    def chat(self, **kwargs):
        self.last_request = kwargs
        return {
            "model": "gemma3:4b",
            "message": {
                "role": "assistant",
                "content": "Latex is documented. [SOURCE 1]",
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 10_000_000,
            "prompt_eval_count": 100,
            "eval_count": 8,
        }


def fake_ocr(_image: Image.Image, *, language: str) -> str:
    assert language == "eng"
    return (
        "Patient ID: SYN-200849\n"
        "Clinical Notes (SOAP)\n"
        "P: Follow-up appointment in 6 months.\n"
        "Medications\nCurrent: Metformin\n"
        "Allergies\nLatex\n"
        "Problems\nActive Diagnosis: Routine exam\n"
    )


def test_ollama_adapter_generates_cited_answer_end_to_end(
    tmp_path: Path,
) -> None:
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

    fake_server = FakeOllamaServer()
    ollama_client = OllamaLLMClient(
        OllamaLLMConfig(model="gemma3:4b"),
        client=fake_server,
    )
    services = RuntimeServices(
        ApiSettings(
            embedding_backend="hash",
            storage_backend="local",
            reranker_backend="deterministic",
            llm_backend="ollama",
            persistence_directory=str(tmp_path / "vectors"),
            registry_path=str(tmp_path / "registry.sqlite3"),
            upload_directory=str(tmp_path / "uploads"),
            log_path=str(tmp_path / "logs/events.jsonl"),
            collection_name="ollama_integration",
            chunk_size=300,
            chunk_overlap=40,
        ),
        ocr_function=fake_ocr,
        llm_client=ollama_client,
    )

    report = services.ingest(samples)
    assert report.failed_files == 0
    assert report.processed_files == 2

    question = "What allergies are documented?"
    retrieved = services.hybrid_retriever.search(
        VectorSearchRequest(
            query=question,
            top_k=10,
            filters=SearchFilters(
                patient_id="SYN-200849"
            ),
        )
    )
    reranked = services.reranker.rerank(
        RerankingRequest(
            query=question,
            passages=retrieved.results,
            top_n=5,
        )
    )
    answer = services.generator.generate(
        question=question,
        passages=reranked.results,
    )
    validation = validate_answer(
        answer,
        expected_patient_id="SYN-200849",
    )

    assert answer.answer == "Latex is documented. [SOURCE 1]"
    assert answer.llm_response.model_name == "ollama:gemma3:4b"
    assert answer.citations
    assert answer.citations[0].patient_id == "SYN-200849"
    assert validation.valid is True

    assert fake_server.last_request is not None
    messages = fake_server.last_request["messages"]
    assert messages[0]["role"] == "system"
    assert "Use only" in messages[0]["content"]
    assert "[SOURCE 1]" in messages[1]["content"]
    assert "Latex" in messages[1]["content"]
