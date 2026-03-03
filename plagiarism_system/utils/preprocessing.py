"""Text preprocessing utilities used across detection engines."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

MULTISPACE_PATTERN = re.compile(r"\s+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def clean_text(text: str) -> str:
    """Normalize text by stripping leading/trailing whitespace and collapsing spaces."""
    normalized = MULTISPACE_PATTERN.sub(" ", (text or "").strip())
    return normalized.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like chunks using punctuation and newlines."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return [chunk.strip() for chunk in SENTENCE_SPLIT_PATTERN.split(cleaned) if chunk and chunk.strip()]


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return TOKEN_PATTERN.findall((text or "").lower())


def generate_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Generate ordered n-grams from a token sequence."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[idx : idx + n]) for idx in range(0, len(tokens) - n + 1)]


def flatten_texts(texts: Iterable[str]) -> str:
    """Join an iterable of texts into a single normalized text block."""
    return clean_text(" ".join(text for text in texts if text))

