"""Document ingestion and collection-statistics endpoints."""

from pathlib import Path
import re
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.api.dependencies import RuntimeServices, get_services
from app.api.schemas import (
    DocumentStatsApiResponse,
    IngestionApiResponse,
    LocalIngestApiRequest,
)
from app.ingestion.pipeline import SUPPORTED_SUFFIXES


router = APIRouter(prefix="/documents", tags=["documents"])
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _to_response(services, report) -> IngestionApiResponse:
    return IngestionApiResponse(
        discovered_files=report.discovered_files,
        processed_files=report.processed_files,
        skipped_files=report.skipped_files,
        failed_files=report.failed_files,
        page_count=report.page_count,
        chunk_count=report.chunk_count,
        indexed_chunk_count=report.indexed_chunk_count,
        collection_count=services.store.count(),
        document_ids=list(report.document_ids),
        warnings=list(report.warnings),
    )


@router.post("/ingest-local", response_model=IngestionApiResponse)
def ingest_local(
    request: LocalIngestApiRequest,
    services: RuntimeServices = Depends(get_services),
) -> IngestionApiResponse:
    """Ingest a trusted server-visible path."""
    try:
        report = services.ingest(request.source_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    services.metrics.increment("documents_ingested_requests_total")
    services.metrics.increment(
        "documents_processed_total",
        report.processed_files,
    )
    services.logger.log(
        "documents_ingested",
        fields={
            "source_type": "local_path",
            "processed_files": report.processed_files,
            "skipped_files": report.skipped_files,
            "failed_files": report.failed_files,
            "indexed_chunks": report.indexed_chunk_count,
        },
    )
    return _to_response(services, report)


@router.post("/upload", response_model=IngestionApiResponse)
async def upload_document(
    file: UploadFile = File(...),
    services: RuntimeServices = Depends(get_services),
) -> IngestionApiResponse:
    """Validate, store, and ingest one uploaded document."""
    original_name = Path(file.filename or "upload").name
    suffix = Path(original_name).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="supported formats are PDF, JPEG, JPG, and PNG",
        )

    content = await file.read(
        services.settings.maximum_upload_bytes + 1
    )
    if len(content) > services.settings.maximum_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="uploaded file exceeds configured size limit",
        )

    safe_stem = _SAFE_NAME_PATTERN.sub(
        "_",
        Path(original_name).stem,
    ).strip("._") or "document"
    destination = (
        Path(services.settings.upload_directory)
        / f"{safe_stem}-{uuid.uuid4().hex[:12]}{suffix}"
    )
    destination.write_bytes(content)

    try:
        report = services.ingest(destination)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"document ingestion failed: {exc}",
        ) from exc
    finally:
        await file.close()

    services.metrics.increment("documents_uploaded_total")
    services.logger.log(
        "document_uploaded",
        fields={
            "suffix": suffix,
            "size_bytes": len(content),
            "processed_files": report.processed_files,
            "failed_files": report.failed_files,
        },
    )
    return _to_response(services, report)


@router.get("/stats", response_model=DocumentStatsApiResponse)
def document_stats(
    services: RuntimeServices = Depends(get_services),
) -> DocumentStatsApiResponse:
    return DocumentStatsApiResponse(
        collection_count=services.store.count(),
        storage_backend=services.store.backend_name,
        embedding_model=services.embedding_provider.model_name,
        upload_directory=str(
            Path(services.settings.upload_directory).resolve()
        ),
    )
