"""Ollama implementation of the project's backend-independent LLM contract.

The client uses Ollama's official Python library but imports it lazily. This
keeps the project's deterministic/offline test mode runnable when the optional
``ollama`` package is not installed.

Pseudo-code for one request:

    validate the configured host and model
    translate the project's system/user prompts into Ollama chat messages
    translate temperature, token limit, stop sequences, and context length
    call Ollama's non-streaming chat endpoint
    extract assistant text from dict-style or object-style responses
    normalize model timing and token metadata
    return the project's shared LLMResponse

The class accepts an injected ``client`` object. Tests use this seam to verify
the real integration contract without requiring a running model server.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlparse

from app.generation.llm_client import LLMClient, LLMRequest, LLMResponse


class OllamaClientProtocol(Protocol):
    """Small subset of the official Ollama client used by this project."""

    def chat(self, **kwargs: Any) -> Any:
        ...

    def list(self) -> Any:
        ...


class OllamaBackendError(RuntimeError):
    """Base error raised by the Ollama adapter."""


class OllamaConfigurationError(OllamaBackendError):
    """Raised when host/model settings are invalid."""


class OllamaUnavailableError(OllamaBackendError):
    """Raised when Ollama cannot be reached or the request fails."""


class OllamaModelNotFoundError(OllamaBackendError):
    """Raised when the configured model is not installed."""


@dataclass(frozen=True)
class OllamaLLMConfig:
    """Runtime settings for local Ollama generation."""

    model: str = "gemma3:4b"
    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0
    keep_alive: str = "5m"
    context_length: int = 8192

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise OllamaConfigurationError(
                "Ollama model must not be blank."
            )
        parsed = urlparse(self.host)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise OllamaConfigurationError(
                "Ollama host must be an http(s) URL, for example "
                "http://127.0.0.1:11434."
            )
        if self.timeout_seconds <= 0:
            raise OllamaConfigurationError(
                "timeout_seconds must be greater than zero."
            )
        if self.context_length <= 0:
            raise OllamaConfigurationError(
                "context_length must be greater than zero."
            )


@dataclass(frozen=True)
class OllamaStatus:
    """Result returned by the explicit server/model readiness check."""

    reachable: bool
    model_available: bool
    host: str
    model: str
    installed_models: tuple[str, ...] = ()
    detail: str | None = None


def _value(container: Any, key: str, default: Any = None) -> Any:
    """Read a field from mapping-style or attribute-style API responses."""
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _model_names(response: Any) -> tuple[str, ...]:
    """Normalize model names from Ollama's list response."""
    models = _value(response, "models", ()) or ()
    names: list[str] = []

    for model in models:
        name = (
            _value(model, "model")
            or _value(model, "name")
            or ""
        )
        cleaned = str(name).strip()
        if cleaned and cleaned not in names:
            names.append(cleaned)

    return tuple(names)


class OllamaLLMClient(LLMClient):
    """Generate citation-grounded answers with a local Ollama model."""

    def __init__(
        self,
        config: OllamaLLMConfig | None = None,
        *,
        client: OllamaClientProtocol | None = None,
    ) -> None:
        self.config = config or OllamaLLMConfig()
        self._client = client or self._create_official_client()

    @property
    def model_name(self) -> str:
        return f"ollama:{self.config.model}"

    @property
    def backend_name(self) -> str:
        return "ollama"

    def _create_official_client(self) -> OllamaClientProtocol:
        """Create the official Python client only when Ollama is selected."""
        try:
            from ollama import Client
        except ImportError as exc:
            raise OllamaConfigurationError(
                "The optional 'ollama' Python package is not installed. "
                "Run: python -m pip install -r requirements-ollama.txt"
            ) from exc

        # The official client forwards extra keyword arguments to httpx.Client.
        return Client(
            host=self.config.host,
            timeout=self.config.timeout_seconds,
        )

    def check_status(self) -> OllamaStatus:
        """Check server reachability and whether the requested model exists.

        Pseudo-code:
            call Ollama list()
            normalize installed model names
            match exact model or its base/latest alias
            return a non-throwing status object
        """
        try:
            response = self._client.list()
        except Exception as exc:
            return OllamaStatus(
                reachable=False,
                model_available=False,
                host=self.config.host,
                model=self.config.model,
                detail=str(exc),
            )

        names = _model_names(response)
        requested = self.config.model
        requested_base = requested.split(":", 1)[0]

        available = any(
            name == requested
            or name == f"{requested}:latest"
            or name.split(":", 1)[0] == requested_base
            for name in names
        )

        return OllamaStatus(
            reachable=True,
            model_available=available,
            host=self.config.host,
            model=requested,
            installed_models=names,
            detail=None if available else (
                f"Model '{requested}' is not installed. "
                f"Run: ollama pull {requested}"
            ),
        )

    def require_ready(self) -> OllamaStatus:
        """Raise a clear error unless both server and model are ready."""
        status = self.check_status()

        if not status.reachable:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {status.host}: "
                f"{status.detail or 'unknown error'}"
            )
        if not status.model_available:
            raise OllamaModelNotFoundError(
                status.detail
                or f"Ollama model '{status.model}' is unavailable."
            )
        return status

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a non-streaming chat response through Ollama."""
        started = perf_counter()

        options: dict[str, Any] = {
            "temperature": request.config.temperature,
            "num_predict": request.config.max_tokens,
            "num_ctx": self.config.context_length,
        }
        if request.config.stop_sequences:
            options["stop"] = list(request.config.stop_sequences)

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": request.system_prompt,
                },
                {
                    "role": "user",
                    "content": request.user_prompt,
                },
            ],
            "stream": False,
            "options": options,
            "keep_alive": self.config.keep_alive,
        }

        try:
            response = self._client.chat(**payload)
        except Exception as exc:
            raise OllamaUnavailableError(
                f"Ollama chat request failed for model "
                f"'{self.config.model}' at {self.config.host}: {exc}"
            ) from exc

        message = _value(response, "message")
        content = str(_value(message, "content", "") or "").strip()

        if not content:
            raise OllamaBackendError(
                "Ollama returned an empty assistant message."
            )

        elapsed_ms = (perf_counter() - started) * 1000
        server_duration_ns = int(
            _value(response, "total_duration", 0) or 0
        )

        return LLMResponse(
            text=content,
            model_name=self.model_name,
            duration_ms=elapsed_ms,
            input_characters=(
                len(request.system_prompt)
                + len(request.user_prompt)
            ),
            output_characters=len(content),
            metadata={
                "backend": "ollama",
                "host": self.config.host,
                "configured_model": self.config.model,
                "response_model": _value(
                    response,
                    "model",
                    self.config.model,
                ),
                "done_reason": _value(response, "done_reason"),
                "prompt_eval_count": int(
                    _value(response, "prompt_eval_count", 0) or 0
                ),
                "eval_count": int(
                    _value(response, "eval_count", 0) or 0
                ),
                "load_duration_ns": int(
                    _value(response, "load_duration", 0) or 0
                ),
                "server_total_duration_ns": server_duration_ns,
                "server_total_duration_ms": (
                    server_duration_ns / 1_000_000
                    if server_duration_ns
                    else None
                ),
            },
        )

    def close(self) -> None:
        """Close the underlying HTTP client when supported."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

