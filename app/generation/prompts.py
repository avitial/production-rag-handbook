"""Grounded RAG prompt templates.

The prompts instruct the model to:

- Use only supplied context
- Cite numbered sources
- Avoid combining patients
- Abstain when explicit evidence is absent
- Distinguish documented facts from inference
- Keep answers concise and auditable
"""

from __future__ import annotations

from dataclasses import dataclass

from app.generation.context_builder import BuiltContext


DEFAULT_SYSTEM_PROMPT = """You are a medical-document question-answering assistant.

Rules:
1. Use only the facts explicitly stated in the provided context.
2. Do not use outside medical knowledge to fill missing details.
3. Do not combine information from different patients.
4. Cite every factual claim with one or more source markers such as [SOURCE 1].
5. If the context does not explicitly support the answer, say:
   "I could not find enough explicit evidence in the provided sources to answer this question."
6. Do not diagnose, recommend treatment, or reinterpret clinical findings.
7. Keep the answer concise.
"""


@dataclass(frozen=True)
class PromptBundle:
    """System and user prompts sent to the model."""

    system_prompt: str
    user_prompt: str


def build_user_prompt(
    *,
    question: str,
    context: BuiltContext,
) -> str:
    """Build a structured prompt with explicit context boundaries.

    Pseudo-code:
        validate question
        add question block
        add instructions about citations and abstention
        add numbered source context
        close context boundary
        request final answer
    """
    if not question.strip():
        raise ValueError("question must not be blank")

    context_text = (
        context.text
        if context.text
        else "(No source passages were retrieved.)"
    )

    return f"""QUESTION:
{question.strip()}

INSTRUCTIONS:
- Answer only from the context below.
- Cite factual statements using [SOURCE N].
- Keep patient information separated.
- If the answer is not explicit, use the required abstention sentence.

BEGIN CONTEXT
{context_text}
END CONTEXT

FINAL ANSWER:
"""


def build_prompt_bundle(
    *,
    question: str,
    context: BuiltContext,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> PromptBundle:
    """Build both prompts and validate non-empty system instructions."""
    if not system_prompt.strip():
        raise ValueError("system_prompt must not be blank")

    return PromptBundle(
        system_prompt=system_prompt.strip(),
        user_prompt=build_user_prompt(
            question=question,
            context=context,
        ),
    )