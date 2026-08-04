"""Dependency-free local LLM substitute for tests and offline demonstrations.

This client does not run a neural language model. It parses the structured
context produced by ``ContextBuilder`` and returns a deterministic,
citation-preserving answer.

It exists so the complete RAG workflow can be tested without API keys, model
downloads, or network access.

Pseudo-code:

    parse QUESTION block
    parse numbered SOURCE blocks
    score sources by overlap with question terms
    detect answer-bearing medical fields
    if no sufficient evidence:
        return abstention
    otherwise:
        extract the most relevant lines
        compose concise answer
        append source citations
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter

from app.generation.llm_client import (
    LLMClient,
    LLMRequest,
    LLMResponse,
)


_SOURCE_PATTERN = re.compile(
    r"\[SOURCE (?P<number>\d+)\]\n"
    r"(?P<header>.*?)\n"
    r"(?P<text>.*?)(?=\n\[SOURCE \d+\]|\nEND CONTEXT|\Z)",
    re.DOTALL,
)

_QUESTION_PATTERN = re.compile(
    r"QUESTION:\n(?P<question>.*?)(?:\n\n|$)",
    re.DOTALL,
)

_TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:[-_/][a-z0-9]+)*",
    re.IGNORECASE,
)

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "in",
    "is", "it", "of", "on", "or", "the", "to", "was", "were",
    "what", "when", "which", "who", "with",
}


@dataclass(frozen=True)
class ParsedSource:
    number: int
    header: str
    text: str


class DeterministicLocalLLMClient(LLMClient):
    """Rule-based grounded generation backend."""

    @property
    def model_name(self) -> str:
        return "deterministic-grounded-generator-v1"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            match.group(0).lower()
            for match in _TOKEN_PATTERN.finditer(text)
            if match.group(0).lower() not in _STOP_WORDS
        ]

    @staticmethod
    def _parse_sources(user_prompt: str) -> list[ParsedSource]:
        return [
            ParsedSource(
                number=int(match.group("number")),
                header=match.group("header").strip(),
                text=match.group("text").strip(),
            )
            for match in _SOURCE_PATTERN.finditer(user_prompt)
        ]

    @staticmethod
    def _parse_question(user_prompt: str) -> str:
        match = _QUESTION_PATTERN.search(user_prompt)
        return match.group("question").strip() if match else ""

    def _source_score(
        self,
        question: str,
        source: ParsedSource,
    ) -> float:
        question_tokens = self._tokens(question)
        source_tokens = self._tokens(source.text)
        source_set = set(source_tokens)

        overlap = sum(
            1 for token in question_tokens if token in source_set
        )

        lowered_question = question.lower()
        lowered_text = source.text.lower()
        label_bonus = 0.0

        mappings = {
            "allerg": ("allergies", "allergy"),
            "medication": ("medications", "medication", "current:"),
            "prescribed": ("medications", "prescription"),
            "follow": ("follow-up", "follow up", "plan"),
            "diagnosis": ("diagnosis", "problems", "assessment"),
            "cardiology": ("cardiology", "referral"),
        }

        for query_term, labels in mappings.items():
            if query_term in lowered_question and any(
                label in lowered_text for label in labels
            ):
                label_bonus += 2.0

        return float(overlap) + label_bonus

    @staticmethod
    def _extract_answer_lines(
        question: str,
        source: ParsedSource,
    ) -> list[str]:
        """Extract likely answer-bearing lines from one source."""
        lines = [
            line.strip()
            for line in source.text.splitlines()
            if line.strip()
        ]
        lowered_question = question.lower()

        keyword_groups = []
        if "allerg" in lowered_question:
            keyword_groups = ["allerg", "latex", "shellfish"]
        elif "medication" in lowered_question or "prescribed" in lowered_question:
            keyword_groups = [
                "medication", "current:", "metformin", "lisinopril"
            ]
        elif "follow" in lowered_question or "appointment" in lowered_question:
            keyword_groups = ["follow-up", "follow up", "appointment", "plan"]
        elif "diagnosis" in lowered_question:
            keyword_groups = [
                "diagnosis", "problems", "hypertension", "routine exam"
            ]
        elif "cardiology" in lowered_question or "referral" in lowered_question:
            keyword_groups = ["cardiology", "referral"]
        else:
            keyword_groups = self_tokens = []

        selected: list[str] = []
        for line in lines:
            lowered = line.lower()
            if any(keyword in lowered for keyword in keyword_groups):
                selected.append(line)

        # Include the line after a section heading such as "Allergies".
        for index, line in enumerate(lines[:-1]):
            lowered = line.lower().rstrip(":")
            if (
                ("allerg" in lowered_question and lowered == "allergies")
                or (
                    ("medication" in lowered_question or "prescribed" in lowered_question)
                    and lowered == "medications"
                )
                or (
                    "diagnosis" in lowered_question
                    and lowered in {"problems", "assessment"}
                )
            ):
                selected.append(lines[index + 1])

        deduplicated: list[str] = []
        for line in selected:
            if line not in deduplicated:
                deduplicated.append(line)
        return deduplicated[:3]

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a deterministic answer from numbered context sources."""
        started = perf_counter()
        question = self._parse_question(request.user_prompt)
        sources = self._parse_sources(request.user_prompt)

        ranked = sorted(
            sources,
            key=lambda source: (
                -self._source_score(question, source),
                source.number,
            ),
        )

        # Qualifying language must be explicitly present in the evidence.
        # A medication list alone does not prove that a drug was prescribed
        # "at discharge", and a follow-up note does not prove a referral.
        lowered_question = question.lower()
        required_phrases: tuple[str, ...] = ()
        if "at discharge" in lowered_question or "discharge" in lowered_question:
            required_phrases = ("discharge",)
        elif "cardiology referral" in lowered_question:
            required_phrases = ("cardiology", "referral")

        if required_phrases and not any(
            all(phrase in source.text.lower() for phrase in required_phrases)
            for source in sources
        ):
            answer = (
                "I could not find enough explicit evidence in the provided "
                "sources to answer this question."
            )
            duration_ms = (perf_counter() - started) * 1000
            return LLMResponse(
                text=answer,
                model_name=self.model_name,
                duration_ms=duration_ms,
                input_characters=(
                    len(request.system_prompt)
                    + len(request.user_prompt)
                ),
                output_characters=len(answer),
                metadata={
                    "backend": "deterministic",
                    "source_count": len(sources),
                    "used_source_numbers": [],
                    "abstention_reason": "missing_qualifying_evidence",
                },
            )

        answer_parts: list[str] = []
        used_sources: list[int] = []

        for source in ranked:
            score = self._source_score(question, source)
            lines = self._extract_answer_lines(question, source)

            if score <= 0 or not lines:
                continue

            answer_parts.extend(lines)
            used_sources.append(source.number)

            # Keep the offline answer concise.
            if len(answer_parts) >= 3:
                break

        if not answer_parts:
            answer = (
                "I could not find enough explicit evidence in the provided "
                "sources to answer this question."
            )
        else:
            cleaned: list[str] = []
            for value in answer_parts:
                if value not in cleaned:
                    cleaned.append(value)

            answer = " ".join(cleaned)
            citations = " ".join(
                f"[SOURCE {number}]"
                for number in sorted(set(used_sources))
            )
            answer = f"{answer} {citations}".strip()

        duration_ms = (perf_counter() - started) * 1000
        return LLMResponse(
            text=answer,
            model_name=self.model_name,
            duration_ms=duration_ms,
            input_characters=(
                len(request.system_prompt)
                + len(request.user_prompt)
            ),
            output_characters=len(answer),
            metadata={
                "backend": "deterministic",
                "source_count": len(sources),
                "used_source_numbers": sorted(set(used_sources)),
            },
        )