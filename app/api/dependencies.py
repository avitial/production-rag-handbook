"""Application settings and shared runtime dependencies.

Pseudo-code:

    read environment variables
    create one embedding provider and persistent store
    create ingestion, vector, BM25, and hybrid retrieval services
    create reranker, grounded generator, confidence policy, logs, and metrics
    cache the runtime so all routes share synchronized indexes

Tests may inject OCR and override ``get_services``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from threading import RLock
from typing import Callable

from app.confidence.policy import ConfidencePolicy
from app.embeddings.sentence_transformer import create_embedding_provider
from app.generation.context_builder import ContextBuilder, ContextBuilderConfig
from app.generation.factory import create_llm_client
from app.generation.llm_client import LLMClient
from app.generation.rag_generator import RAGGenerator
from app.ingestion.pipeline import IngestionConfig, LocalIngestionPipeline
from app.observability.logger import StructuredLogger
from app.observability.metrics import MetricsCollector
from app.reranking.cross_encoder import create_reranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import (
    HybridRetrievalConfig,
    HybridRetriever,
)
from app.retrieval.vector_retriever import VectorRetriever
from app.storage.chroma_store import ChromaStore
from app.storage.document_registry import DocumentRegistry


@dataclass(frozen=True)
class ApiSettings:
    service_name: str = "Medical Document Assistant"
    service_version: str = "1.0.0-day14"
    embedding_backend: str = "auto"
    storage_backend: str = "auto"
    reranker_backend: str = "auto"
    llm_backend: str = "deterministic"
    ollama_model: str = "gemma3:4b"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = 120.0
    ollama_keep_alive: str = "5m"
    ollama_context_length: int = 8192
    persistence_directory: str = "./runtime/chroma"
    registry_path: str = "./runtime/document-registry.sqlite3"
    upload_directory: str = "./runtime/uploads"
    log_path: str = "./runtime/logs/api-events.jsonl"
    collection_name: str = "medical_documents_api"
    chunk_size: int = 800
    chunk_overlap: int = 150
    maximum_upload_bytes: int = 15 * 1024 * 1024
    maximum_context_characters: int = 6000
    maximum_context_sources: int = 6

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        return cls(
            service_name=os.getenv(
                "MDA_SERVICE_NAME",
                "Medical Document Assistant",
            ),
            embedding_backend=os.getenv(
                "MDA_EMBEDDING_BACKEND",
                "auto",
            ),
            storage_backend=os.getenv(
                "MDA_STORAGE_BACKEND",
                "auto",
            ),
            reranker_backend=os.getenv(
                "MDA_RERANKER_BACKEND",
                "auto",
            ),
            llm_backend=os.getenv(
                "MDA_LLM_BACKEND",
                "deterministic",
            ),
            ollama_model=os.getenv(
                "MDA_OLLAMA_MODEL",
                os.getenv("OLLAMA_MODEL", "gemma3:4b"),
            ),
            ollama_host=os.getenv(
                "MDA_OLLAMA_HOST",
                os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ),
            ollama_timeout_seconds=float(
                os.getenv("MDA_OLLAMA_TIMEOUT_SECONDS", "120")
            ),
            ollama_keep_alive=os.getenv(
                "MDA_OLLAMA_KEEP_ALIVE",
                "5m",
            ),
            ollama_context_length=int(
                os.getenv("MDA_OLLAMA_CONTEXT_LENGTH", "8192")
            ),
            persistence_directory=os.getenv(
                "MDA_CHROMA_DIR",
                "./runtime/chroma",
            ),
            registry_path=os.getenv(
                "MDA_REGISTRY_PATH",
                "./runtime/document-registry.sqlite3",
            ),
            upload_directory=os.getenv(
                "MDA_UPLOAD_DIR",
                "./runtime/uploads",
            ),
            log_path=os.getenv(
                "MDA_LOG_PATH",
                "./runtime/logs/api-events.jsonl",
            ),
            collection_name=os.getenv(
                "MDA_COLLECTION_NAME",
                "medical_documents_api",
            ),
            chunk_size=int(os.getenv("MDA_CHUNK_SIZE", "800")),
            chunk_overlap=int(
                os.getenv("MDA_CHUNK_OVERLAP", "150")
            ),
            maximum_upload_bytes=int(
                os.getenv(
                    "MDA_MAX_UPLOAD_BYTES",
                    str(15 * 1024 * 1024),
                )
            ),
            maximum_context_characters=int(
                os.getenv("MDA_CONTEXT_CHARACTERS", "6000")
            ),
            maximum_context_sources=int(
                os.getenv("MDA_CONTEXT_SOURCES", "6")
            ),
        )


class RuntimeServices:
    """Shared and synchronized service graph for the API."""

    def __init__(
        self,
        settings: ApiSettings,
        *,
        ocr_function: Callable | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.lock = RLock()

        Path(settings.upload_directory).mkdir(parents=True, exist_ok=True)
        Path(settings.persistence_directory).mkdir(parents=True, exist_ok=True)
        Path(settings.registry_path).parent.mkdir(parents=True, exist_ok=True)

        self.embedding_provider = create_embedding_provider(
            settings.embedding_backend
        )
        self.store = ChromaStore(
            embedding_provider=self.embedding_provider,
            collection_name=settings.collection_name,
            persistence_directory=settings.persistence_directory,
            backend=settings.storage_backend,
        )
        self.registry = DocumentRegistry(settings.registry_path)
        self.ingestion = LocalIngestionPipeline(
            chroma_store=self.store,
            registry=self.registry,
            config=IngestionConfig(
                max_characters=settings.chunk_size,
                overlap_characters=settings.chunk_overlap,
            ),
            ocr_function=ocr_function,
        )
        self.vector_retriever = VectorRetriever(self.store)
        self.bm25_retriever = BM25Retriever(self.store)
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            config=HybridRetrievalConfig(
                vector_top_k=15,
                keyword_top_k=15,
                fusion_constant=60,
            ),
        )
        self.reranker = create_reranker(settings.reranker_backend)
        selected_llm_client = llm_client or create_llm_client(
            settings.llm_backend,
            ollama_model=settings.ollama_model,
            ollama_host=settings.ollama_host,
            ollama_timeout_seconds=settings.ollama_timeout_seconds,
            ollama_keep_alive=settings.ollama_keep_alive,
            ollama_context_length=settings.ollama_context_length,
        )
        self.generator = RAGGenerator(
            llm_client=selected_llm_client,
            context_builder=ContextBuilder(
                ContextBuilderConfig(
                    maximum_characters=settings.maximum_context_characters,
                    maximum_sources=settings.maximum_context_sources,
                )
            ),
        )
        self.confidence_policy = ConfidencePolicy()
        self.logger = StructuredLogger(settings.log_path)
        self.metrics = MetricsCollector()
        self.rebuild_keyword_index()

    def rebuild_keyword_index(self) -> int:
        with self.lock:
            return self.bm25_retriever.rebuild()

    def ingest(self, source_path: str | Path):
        with self.lock:
            report = self.ingestion.ingest(source_path)
            if report.failed_files == 0:
                self.bm25_retriever.rebuild()
            return report


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings.from_environment()


@lru_cache(maxsize=1)
def get_services() -> RuntimeServices:
    return RuntimeServices(get_settings())


def reset_dependency_caches() -> None:
    get_services.cache_clear()
    get_settings.cache_clear()