"""Factory for selecting deterministic or Ollama answer generation.

Pseudo-code:

    normalize backend name
    deterministic -> create dependency-free test/demo generator
    ollama       -> create Ollama adapter with host/model settings
    auto         -> try Ollama only when it is reachable and model exists;
                    otherwise return deterministic client
"""

from __future__ import annotations

from app.generation.llm_client import LLMClient
from app.generation.local_llm_client import DeterministicLocalLLMClient
from app.generation.ollama_llm_client import (
    OllamaLLMClient,
    OllamaLLMConfig,
)


def create_llm_client(
    backend: str = "deterministic",
    *,
    ollama_model: str = "gemma3:4b",
    ollama_host: str = "http://127.0.0.1:11434",
    ollama_timeout_seconds: float = 120.0,
    ollama_keep_alive: str = "5m",
    ollama_context_length: int = 8192,
) -> LLMClient:
    """Create the configured generation backend."""
    normalized = backend.strip().lower()

    if normalized in {"deterministic", "offline", "test"}:
        return DeterministicLocalLLMClient()

    config = OllamaLLMConfig(
        model=ollama_model,
        host=ollama_host,
        timeout_seconds=ollama_timeout_seconds,
        keep_alive=ollama_keep_alive,
        context_length=ollama_context_length,
    )

    if normalized == "ollama":
        return OllamaLLMClient(config)

    if normalized == "auto":
        try:
            client = OllamaLLMClient(config)
            status = client.check_status()
            if status.reachable and status.model_available:
                return client
        except Exception:
            pass
        return DeterministicLocalLLMClient()

    raise ValueError(
        "LLM backend must be deterministic, ollama, or auto."
    )

