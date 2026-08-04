"""Unit tests for grounded RAG prompt construction."""

from app.generation.context_builder import BuiltContext
from app.generation.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_prompt_bundle,
    build_user_prompt,
)


def sample_context() -> BuiltContext:
    return BuiltContext(
        text=(
            "[SOURCE 1]\n"
            "File: sample.pdf\n"
            "Page: 1\n"
            "Allergies\nLatex"
        ),
        sources=(),
        included_characters=65,
        omitted_source_count=0,
        truncated=False,
    )


def test_system_prompt_requires_grounding_and_citations() -> None:
    lowered = DEFAULT_SYSTEM_PROMPT.lower()

    assert "use only" in lowered
    assert "cite" in lowered
    assert "do not combine" in lowered
    assert "could not find enough explicit evidence" in lowered


def test_user_prompt_contains_question_and_context_boundaries() -> None:
    prompt = build_user_prompt(
        question="What allergies are documented?",
        context=sample_context(),
    )

    assert "QUESTION:" in prompt
    assert "What allergies are documented?" in prompt
    assert "BEGIN CONTEXT" in prompt
    assert "[SOURCE 1]" in prompt
    assert "END CONTEXT" in prompt
    assert "FINAL ANSWER:" in prompt


def test_empty_context_is_explicit() -> None:
    context = BuiltContext(
        text="",
        sources=(),
        included_characters=0,
        omitted_source_count=0,
        truncated=False,
    )

    prompt = build_user_prompt(
        question="Which patient had a cardiology referral?",
        context=context,
    )

    assert "No source passages were retrieved" in prompt


def test_prompt_bundle_preserves_custom_system_prompt() -> None:
    bundle = build_prompt_bundle(
        question="What is documented?",
        context=sample_context(),
        system_prompt="Use only supplied evidence.",
    )

    assert bundle.system_prompt == "Use only supplied evidence."
    assert "What is documented?" in bundle.user_prompt