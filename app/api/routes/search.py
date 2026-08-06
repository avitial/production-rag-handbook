"""Metadata-filtered hybrid-search endpoint."""

from time import perf_counter

from fastapi import APIRouter, Depends

from app.api.dependencies import RuntimeServices, get_services
from app.api.schemas import (
    SearchApiRequest,
    SearchApiResponse,
    SearchResultApi,
)
from app.retrieval.models import SearchFilters, VectorSearchRequest


router = APIRouter(prefix="/search", tags=["search"])


def _filters(values) -> SearchFilters:
    return SearchFilters(
        patient_id=values.patient_id,
        document_id=values.document_id,
        filename=values.filename,
        source_format=values.source_format,
        page_number=values.page_number,
    )


@router.post("", response_model=SearchApiResponse)
def search(
    request: SearchApiRequest,
    services: RuntimeServices = Depends(get_services),
) -> SearchApiResponse:
    started = perf_counter()
    response = services.hybrid_retriever.search(
        VectorSearchRequest(
            query=request.query,
            top_k=request.top_k,
            filters=_filters(request.filters),
        )
    )
    duration_ms = (perf_counter() - started) * 1000

    services.metrics.increment("search_requests_total")
    services.metrics.observe_latency("api_search_ms", duration_ms)
    services.logger.log(
        "search_completed",
        fields={
            "result_count": len(response.results),
            "patient_filter_applied": bool(request.filters.patient_id),
            "duration_ms": duration_ms,
        },
    )

    return SearchApiResponse(
        query=request.query,
        result_count=len(response.results),
        collection_count=response.diagnostics.collection_count,
        retrieval_model=response.diagnostics.embedding_model,
        applied_filter=response.diagnostics.where_filter,
        results=[
            SearchResultApi(
                rank=item.rank,
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                filename=item.filename,
                page_number=item.page_number,
                section=item.section,
                patient_id=item.patient_id,
                text=item.text,
                score=item.similarity,
                citation_label=item.citation_label(),
            )
            for item in response.results
        ],
    )
