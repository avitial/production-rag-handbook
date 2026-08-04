"""Unit tests for medical-aware tokenization."""
import sys
import os

# Add the absolute path of the external folder
sys.path.append(os.path.abspath("/home/avitial/workspace/RAG/production-rag-handbook"))

from app.retrieval.tokenizer import (
    MedicalTokenizer,
    TokenizerConfig,
    tokenize,
)


def test_preserves_medical_identifiers_and_values() -> None:
    tokens = tokenize(
        "Patient ID SYN-200849, HbA1c 6.8%, "
        "LDL 92 mg/dL, follow-up 2026-07-20."
    )

    assert "syn-200849" in tokens
    assert "hba1c" in tokens
    assert "6.8%" in tokens
    assert "ldl" in tokens
    assert "92" in tokens
    assert "mg/dl" in tokens
    assert "follow-up" in tokens
    assert "2026-07-20" in tokens


def test_removes_common_stop_words_by_default() -> None:
    tokens = tokenize(
        "What are the allergies documented for the patient?"
    )

    assert "what" not in tokens
    assert "the" not in tokens
    assert "allergies" in tokens
    assert "documented" in tokens
    assert "patient" in tokens


def test_stop_word_removal_can_be_disabled() -> None:
    tokenizer = MedicalTokenizer(
        TokenizerConfig(remove_stop_words=False)
    )

    tokens = tokenizer.tokenize("What is the diagnosis?")

    assert "what" in tokens
    assert "is" in tokens
    assert "the" in tokens
    assert "diagnosis" in tokens


def test_unicode_is_normalized() -> None:
    tokens = tokenize("Ｍｅｔｆｏｒｍｉｎ １０mg")

    assert "metformin" in tokens
    assert "10mg" in tokens


def test_tokenization_is_deterministic() -> None:
    text = "Allergy: Shellfish; Medication: Metformin."

    assert tokenize(text) == tokenize(text)
