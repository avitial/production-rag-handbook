"""Abstract LLM client contracts for grounded RAG generation.

The rest of the application depends on this interface rather than on a
specific hosted or local model.

Pseudo-code:

    receive system prompt
    receive user prompt
    receive generation configuration
    invoke concrete model backend
    return text plus model and timing metadata
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Model-generation controls shared across client implementations."""

    temperature: float = 0.0
    max_tokens: int = 600
    stop_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")


@dataclass(frozen=True)
class LLMRequest:
    """One model request."""

    system_prompt: str
    user_prompt: str
    config: GenerationConfig = field(
        default_factory=GenerationConfig
    )

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be blank")
        if not self.user_prompt.strip():
            raise ValueError("user_prompt must not be blank")


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response from any model backend."""

    text: str
    model_name: str
    duration_ms: float
    input_characters: int
    output_characters: int
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Interface implemented by hosted and local language models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model name for logs and evaluation."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one answer from prompts."""