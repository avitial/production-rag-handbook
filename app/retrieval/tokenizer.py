"""Medical-aware tokenizer used by the BM25 keyword retriever.

The tokenizer aims to preserve exact values that matter in medical records:

- Patient IDs: SYN-200849
- Dosages: 10mg, 10 mg
- Laboratory abbreviations: HbA1c, LDL
- Dates: 2026-07-20
- Hyphenated terms: follow-up, COVID-19
- Decimal values: 6.8

Pseudo-code:

    normalize Unicode
    lowercase text
    find tokens with a medical-friendly regular expression
    optionally remove stop words
    optionally add normalized dosage variants
    return token list
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata


DEFAULT_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "had", "has", "have", "in", "is", "it", "of", "on",
        "or", "that", "the", "to", "was", "were", "what", "when",
        "which", "who", "with",
    }
)

TOKEN_PATTERN = re.compile(
    r"""
    [a-z0-9]+(?:[-_/][a-z0-9]+)+     # SYN-200849, follow-up, COVID-19
    |
    \d+(?:\.\d+)?(?:mg|mcg|g|ml|mmhg|mg/dl|%)  # 10mg, 6.8%, 92mg/dl
    |
    [a-z]+\d+[a-z0-9]*               # hba1c, z00
    |
    \d{4}-\d{2}-\d{2}                # ISO date
    |
    \d+(?:\.\d+)?                    # numbers and decimals
    |
    [a-z]+                            # ordinary words
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for deterministic keyword tokenization."""

    lowercase: bool = True
    remove_stop_words: bool = True
    stop_words: frozenset[str] = field(
        default_factory=lambda: DEFAULT_STOP_WORDS
    )
    minimum_token_length: int = 1

    def __post_init__(self) -> None:
        if self.minimum_token_length <= 0:
            raise ValueError(
                "minimum_token_length must be greater than zero"
            )


class MedicalTokenizer:
    """Tokenize clinical text and exact identifiers consistently."""

    def __init__(
        self,
        config: TokenizerConfig | None = None,
    ) -> None:
        self.config = config or TokenizerConfig()

    def normalize(self, text: str) -> str:
        """Normalize Unicode and whitespace before tokenization."""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        if self.config.lowercase:
            normalized = normalized.lower()
        return normalized

    def tokenize(self, text: str) -> list[str]:
        """Return stable tokens for BM25 indexing and querying.

        Pseudo-code:
            normalize input
            regex-find candidate tokens
            remove short tokens
            remove configured stop words
            append remaining tokens in original order
        """
        normalized = self.normalize(text)
        output: list[str] = []

        for match in TOKEN_PATTERN.finditer(normalized):
            token = match.group(0)
            if len(token) < self.config.minimum_token_length:
                continue
            if (
                self.config.remove_stop_words
                and token in self.config.stop_words
            ):
                continue
            output.append(token)

        return output


def tokenize(
    text: str,
    *,
    config: TokenizerConfig | None = None,
) -> list[str]:
    """Convenience function for one-off tokenization."""
    return MedicalTokenizer(config).tokenize(text)
