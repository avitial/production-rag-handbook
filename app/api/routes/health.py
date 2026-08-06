"""Health and readiness endpoints."""

from fastapi import APIRouter, Depends

from app.api.dependencies import RuntimeServices, get_services
from app.api.schemas import HealthApiResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthApiResponse)
def health(
    services: RuntimeServices = Depends(get_services),
) -> HealthApiResponse:
    return HealthApiResponse(
        status="ok",
        service=services.settings.service_name,
        version=services.settings.service_version,
        storage_backend=services.store.backend_name,
        embedding_model=services.embedding_provider.model_name,
        reranker_model=services.reranker.model_name,
        generator_model=services.generator.llm_client.model_name,
        collection_count=services.store.count(),
    )


@router.get("/ready", response_model=HealthApiResponse)
def ready(
    services: RuntimeServices = Depends(get_services),
) -> HealthApiResponse:
    return health(services)
