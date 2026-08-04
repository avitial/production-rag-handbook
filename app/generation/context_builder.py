"""Build bounded, citation-ready context from reranked passages.

The context builder is responsible for:

- Preserving source provenance
- Assigning stable source numbers
- Enforcing a character budget
- Avoiding duplicate chunks
- Preventing partial passages unless explicitly allowed
- Returning a citation map for post-generation validation

Pseudo-code:

    receive reranked passages
    deduplicate by chunk ID
    for each passage in reranked order:
        build source header
        calculate block size
        if block fits:
            add complete block
        else:
            stop or truncate according to configuration
    return context text and source mapping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from app.reranking.base import RerankedPassage


@dataclass(frozen=True)
class ContextBuilderConfig:
    """Controls context size and formatting."""

    maximum_characters: int = 6000
    maximum_sources: int = 6
    allow_truncation: bool = False
    include_scores: bool = False

    def __post_init__(self) -> None:
        if self.maximum_characters <= 0:
            raise ValueError(
                "maximum_characters must be greater than zero"
            )
        if self.maximum_sources <= 0:
            raise ValueError(
                "maximum_sources must be greater than zero"
            )


@dataclass(frozen=True)
class ContextSource:
    """One source included in the final prompt context."""

    source_number: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section: str | None
    patient_id: str | None
    text: str
    citation_label: str
    rerank_score: float


@dataclass(frozen=True)
class BuiltContext:
    """Complete context plus provenance and budget diagnostics."""

    text: str
    sources: tuple[ContextSource, ...]
    included_characters: int
    omitted_source_count: int
    truncated: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_map(self) -> dict[int, ContextSource]:
        return {
            source.source_number: source
            for source in self.sources
        }


class ContextBuilder:
    """Create structured context for grounded generation."""

    def __init__(
        self,
        config: ContextBuilderConfig | None = None,
    ) -> None:
        self.config = config or ContextBuilderConfig()

    def _header(
        self,
        source_number: int,
        passage: RerankedPassage,
    ) -> str:
        source = passage.passage
        parts = [
            f"[SOURCE {source_number}]",
            f"File: {source.filename}",
            f"Page: {source.page_number}",
        ]

        if source.section:
            parts.append(f"Section: {source.section}")
        if source.patient_id:
            parts.append(f"Patient ID: {source.patient_id}")
        if self.config.include_scores:
            parts.append(
                f"Reranker score: {passage.rerank_score:.6f}"
            )

        return "\n".join(parts)

    def build(
        self,
        passages: Iterable[RerankedPassage],
    ) -> BuiltContext:
        """Build context in the current reranked order."""
        unique: list[RerankedPassage] = []
        seen_chunk_ids: set[str] = set()

        for passage in passages:
            chunk_id = passage.passage.chunk_id
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            unique.append(passage)

        blocks: list[str] = []
        sources: list[ContextSource] = []
        used_characters = 0
        truncated = False

        for passage in unique[:self.config.maximum_sources]:
            source_number = len(sources) + 1
            header = self._header(source_number, passage)
            source_text = passage.passage.text.strip()
            block = f"{header}\n{source_text}\n"
            separator_size = 1 if blocks else 0
            projected = (
                used_characters
                + separator_size
                + len(block)
            )

            if projected <= self.config.maximum_characters:
                blocks.append(block)
                used_characters = projected
                sources.append(
                    ContextSource(
                        source_number=source_number,
                        chunk_id=passage.passage.chunk_id,
                        document_id=passage.passage.document_id,
                        filename=passage.passage.filename,
                        page_number=passage.passage.page_number,
                        section=passage.passage.section,
                        patient_id=passage.passage.patient_id,
                        text=source_text,
                        citation_label=(
                            passage.passage.citation_label()
                        ),
                        rerank_score=passage.rerank_score,
                    )
                )
                continue

            if not self.config.allow_truncation:
                break

            remaining = (
                self.config.maximum_characters
                - used_characters
                - separator_size
                - len(header)
                - 2
            )
            if remaining <= 20:
                break

            truncated_text = source_text[:remaining].rstrip()
            block = f"{header}\n{truncated_text}\n"
            blocks.append(block)
            used_characters += separator_size + len(block)
            truncated = True
            sources.append(
                ContextSource(
                    source_number=source_number,
                    chunk_id=passage.passage.chunk_id,
                    document_id=passage.passage.document_id,
                    filename=passage.passage.filename,
                    page_number=passage.passage.page_number,
                    section=passage.passage.section,
                    patient_id=passage.passage.patient_id,
                    text=truncated_text,
                    citation_label=(
                        passage.passage.citation_label()
                    ),
                    rerank_score=passage.rerank_score,
                )
            )
            break

        omitted = max(0, len(unique) - len(sources))
        return BuiltContext(
            text="\n".join(blocks).strip(),
            sources=tuple(sources),
            included_characters=len(
                "\n".join(blocks).strip()
            ),
            omitted_source_count=omitted,
            truncated=truncated,
            metadata={
                "maximum_characters": (
                    self.config.maximum_characters
                ),
                "maximum_sources": self.config.maximum_sources,
            },
        )