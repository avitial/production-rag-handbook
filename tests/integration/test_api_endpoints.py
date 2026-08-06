"""End-to-end FastAPI tests for Day 14."""

from pathlib import Path
import shutil

from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import ApiSettings, RuntimeServices, get_services
from app.api.main import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HANDWRITTEN = (
    PROJECT_ROOT
    / "data/samples/handwritten/SYN-200849_handwritten.png"
)
NATIVE_PDF = (
    PROJECT_ROOT
    / "data/samples/native-pdf/SYN-200989.pdf"
)


def fake_ocr(_image: Image.Image, *, language: str) -> str:
    assert language == "eng"
    return (
        "*** SYNTHETIC MEDICAL RECORD ***\n"
        "Patient Demographics\n"
        "Patient ID: SYN-200849\n"
        "Clinical Notes (SOAP)\n"
        "P: Follow-up appointment in 6 months.\n"
        "Medications\n"
        "Current: Metformin\n"
        "Allergies\n"
        "Latex\n"
        "Problems\n"
        "Active Diagnosis: Routine exam\n"
    )


def build_client(tmp_path: Path):
    services = RuntimeServices(
        ApiSettings(
            embedding_backend="hash",
            storage_backend="local",
            reranker_backend="deterministic",
            persistence_directory=str(tmp_path / "vectors"),
            registry_path=str(tmp_path / "registry.sqlite3"),
            upload_directory=str(tmp_path / "uploads"),
            log_path=str(tmp_path / "logs/events.jsonl"),
            collection_name="day14_api_tests",
            chunk_size=300,
            chunk_overlap=40,
        ),
        ocr_function=fake_ocr,
    )
    app = create_app()
    app.dependency_overrides[get_services] = lambda: services
    return TestClient(app), services


def test_complete_api_flow(tmp_path: Path) -> None:
    client, _services = build_client(tmp_path)

    assert client.get("/health").json()["status"] == "ok"

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

    ingestion = client.post(
        "/documents/ingest-local",
        json={"source_path": str(samples)},
    )
    assert ingestion.status_code == 200, ingestion.text
    assert ingestion.json()["processed_files"] == 2
    assert ingestion.json()["failed_files"] == 0

    search = client.post(
        "/search",
        json={
            "query": "Latex allergy",
            "filters": {"patient_id": "SYN-200849"},
            "top_k": 5,
        },
    )
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert results
    assert all(x["patient_id"] == "SYN-200849" for x in results)
    assert any("Latex" in x["text"] for x in results)

    question = client.post(
        "/questions",
        json={
            "question": "What allergies are documented?",
            "filters": {"patient_id": "SYN-200849"},
            "candidate_k": 10,
            "final_k": 5,
        },
    )
    assert question.status_code == 200, question.text
    body = question.json()
    assert body["decision"] == "accept"
    assert "Latex" in body["answer"]
    assert body["citations"]
    assert body["validation"]["citation_valid"] is True

    unsupported = client.post(
        "/questions",
        json={
            "question": "Which patient had a cardiology referral?",
            "candidate_k": 10,
            "final_k": 5,
        },
    )
    assert unsupported.status_code == 200
    assert unsupported.json()["decision"] == "abstain"


def test_upload_png_and_reject_unsupported_file(
    tmp_path: Path,
) -> None:
    client, _services = build_client(tmp_path)

    with HANDWRITTEN.open("rb") as handle:
        uploaded = client.post(
            "/documents/upload",
            files={
                "file": (
                    "SYN-200849_handwritten.png",
                    handle,
                    "image/png",
                )
            },
        )

    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()["processed_files"] == 1
    assert uploaded.json()["failed_files"] == 0

    rejected = client.post(
        "/documents/upload",
        files={
            "file": (
                "unsupported.exe",
                b"not a supported document",
                "application/octet-stream",
            )
        },
    )
    assert rejected.status_code == 415
