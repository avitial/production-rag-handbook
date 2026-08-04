"""Hybrid retrieval combining vector search, BM25, and RRF."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.fusion import (
    as_ranked_passages,
    reciprocal_rank_fusion,
)
from app.retrieval.models import (
    RetrievalDiagnostics,
    VectorSearchRequest,
    VectorSearchResponse,
)
from app.retrieval.vector_retriever import VectorRetriever


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Candidate and fusion settings."""

    vector_top_k: int = 15
    keyword_top_k: int = 15
    fusion_constant: int = 60

    def __post_init__(self) -> None:
        if self.vector_top_k <= 0:
            raise ValueError("vector_top_k must be greater than zero")
        if self.keyword_top_k <= 0:
            raise ValueError("keyword_top_k must be greater than zero")
        if self.fusion_constant <= 0:
            raise ValueError(
                "fusion_constant must be greater than zero"
            )


class HybridRetriever:
    """Retrieve through both semantic and lexical search."""

    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.config = config or HybridRetrievalConfig()

    def rebuild_keyword_index(self) -> int:
        """Rebuild BM25 after local documents have been ingested."""
        return self.bm25_retriever.rebuild()

    def search(
        self,
        request: VectorSearchRequest,
    ) -> VectorSearchResponse:
        """Run vector and keyword search, then fuse ranks.

        Pseudo-code:
            create vector request with vector candidate count
            create keyword request with keyword candidate count
            execute both using the same metadata filters
            fuse by chunk ID using RRF
            truncate to caller's requested top-k
            return shared typed response
        """
        vector_response = self.vector_retriever.search(
            VectorSearchRequest(
                query=request.query,
                top_k=self.config.vector_top_k,
                filters=request.filters,
            )
        )
        keyword_response = self.bm25_retriever.search(
            VectorSearchRequest(
                query=request.query,
                top_k=self.config.keyword_top_k,
                filters=request.filters,
            )
        )

        fused = reciprocal_rank_fusion(
            {
                "vector": vector_response.results,
                "bm25": keyword_response.results,
            },
            fusion_constant=self.config.fusion_constant,
            top_n=request.top_k,
        )
        results = as_ranked_passages(fused)

        return VectorSearchResponse(
            query=request.query,
            filters=request.filters,
            results=results,
            diagnostics=RetrievalDiagnostics(
                collection_count=(
                    vector_response.diagnostics.collection_count
                ),
                requested_top_k=request.top_k,
                returned_count=len(results),
                where_filter=(
                    vector_response.diagnostics.where_filter
                ),
                embedding_model=(
                    f"hybrid:"
                    f"{vector_response.diagnostics.embedding_model}"
                    f"+bm25"
                ),
            ),
        )
