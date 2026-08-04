"""Orchestrate grounded answer generation from reranked passages.

Responsibilities:

- Build bounded context
- Build system and user prompts
- Invoke an LLM client
- Validate numbered citations
- Map source numbers back to files and pages
- Return structured generation diagnostics

Pseudo-code:

    context = context_builder.build(reranked passages)
    prompts = build_prompt_bundle(question, context)
    llm_response = llm_client.generate(prompts)
    parse [SOURCE N] citations
    reject citations not present in context
    determine whether answer abstained
    return answer, citations, prompts, and diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from time import perf_counter
from typing import Any, Iterable

from app.generation.context_builder import (
    BuiltContext,
    ContextBuilder,
    ContextSource,
)
from app.generation.llm_client import (
    GenerationConfig,
    LLMClient,
    LLMRequest,
    LLMResponse,
)
from app.generation.prompts import (
    PromptBundle,
    build_prompt_bundle,
)
from app.reranking.base import RerankedPassage


_CITATION_PATTERN = re.compile(
    r"\[SOURCE\s+(\d+)\]",
    re.IGNORECASE,
)

_ABSTENTION_TEXT = (
    "I could not find enough explicit evidence in the provided "
    "sources to answer this question."
)


@dataclass(frozen=True)
class AnswerCitation:
    """Resolved citation included in the generated answer."""

    source_number: int
    chunk_id: str
    filename: str
    page_number: int
    section: str | None
    patient_id: str | None
    citation_label: str


@dataclass(frozen=True)
class RAGGenerationDiagnostics:
    """Generation-stage operational and validation information."""

    llm_model: str
    context_source_count: int
    context_characters: int
    omitted_source_count: int
    context_truncated: bool
    citation_count: int
    invalid_citation_numbers: tuple[int, ...]
    abstained: bool
    duration_ms: float


@dataclass(frozen=True)
class RAGAnswer:
    """Structured result returned by the RAG generator."""

    question: str
    answer: str
    citations: tuple[AnswerCitation, ...]
    context: BuiltContext
    prompts: PromptBundle
    llm_response: LLMResponse
    diagnostics: RAGGenerationDiagnostics
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGGenerator:
    """Generate citation-grounded answers."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        context_builder: ContextBuilder | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.context_builder = (
            context_builder or ContextBuilder()
        )
        self.generation_config = (
            generation_config or GenerationConfig()
        )

    @staticmethod
    def _citation_numbers(answer: str) -> list[int]:
        """Parse source markers in first-occurrence order."""
        output: list[int] = []
        for match in _CITATION_PATTERN.finditer(answer):
            value = int(match.group(1))
            if value not in output:
                output.append(value)
        return output

    @staticmethod
    def _resolve_citation(
        source: ContextSource,
    ) -> AnswerCitation:
        return AnswerCitation(
            source_number=source.source_number,
            chunk_id=source.chunk_id,
            filename=source.filename,
            page_number=source.page_number,
            section=source.section,
            patient_id=source.patient_id,
            citation_label=source.citation_label,
        )

    def generate(
        self,
        *,
        question: str,
        passages: Iterable[RerankedPassage],
    ) -> RAGAnswer:
        """Generate one answer from reranked passages."""
        started = perf_counter()

        context = self.context_builder.build(passages)
        prompts = build_prompt_bundle(
            question=question,
            context=context,
        )
        llm_response = self.llm_client.generate(
            LLMRequest(
                system_prompt=prompts.system_prompt,
                user_prompt=prompts.user_prompt,
                config=self.generation_config,
            )
        )

        citation_numbers = self._citation_numbers(
            llm_response.text
        )
        citation_map = context.citation_map

        invalid = tuple(
            number
            for number in citation_numbers
            if number not in citation_map
        )
        citations = tuple(
            self._resolve_citation(citation_map[number])
            for number in citation_numbers
            if number in citation_map
        )

        answer = llm_response.text.strip()
        abstained = (
            _ABSTENTION_TEXT.lower() in answer.lower()
            or not context.sources
        )

        duration_ms = (perf_counter() - started) * 1000

        return RAGAnswer(
            question=question,
            answer=answer,
            citations=citations,
            context=context,
            prompts=prompts,
            llm_response=llm_response,
            diagnostics=RAGGenerationDiagnostics(
                llm_model=llm_response.model_name,
                context_source_count=len(context.sources),
                context_characters=(
                    context.included_characters
                ),
                omitted_source_count=(
                    context.omitted_source_count
                ),
                context_truncated=context.truncated,
                citation_count=len(citations),
                invalid_citation_numbers=invalid,
                abstained=abstained,
                duration_ms=duration_ms,
            ),
        )