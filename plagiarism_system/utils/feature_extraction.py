"""Common feature extraction helpers for stylometry and AI detection."""

from __future__ import annotations

import math
import string
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def safe_div(numerator: float, denominator: float) -> float:
    """Safely divide numbers and return zero when denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def lexical_diversity(tokens: Sequence[str]) -> float:
    """Compute lexical diversity as unique token ratio."""
    if not tokens:
        return 0.0
    return safe_div(len(set(tokens)), len(tokens))


def hapax_legomena_ratio(tokens: Sequence[str]) -> float:
    """Compute hapax legomena ratio (tokens appearing exactly once)."""
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    hapax_count = sum(1 for value in counts.values() if value == 1)
    return safe_div(hapax_count, len(tokens))


def sentence_length_statistics(sentences: Sequence[str], tokenizer) -> Dict[str, float]:
    """Compute average and variance of sentence lengths using a tokenizer callback."""
    lengths = [len(tokenizer(sentence)) for sentence in sentences if sentence.strip()]
    lengths = [length for length in lengths if length > 0]
    if not lengths:
        return {"avg_sentence_length": 0.0, "sentence_length_variance": 0.0}
    values = np.array(lengths, dtype=float)
    return {
        "avg_sentence_length": float(values.mean()),
        "sentence_length_variance": float(values.var()),
    }


def punctuation_pattern_frequency(text: str) -> Dict[str, float]:
    """Compute normalized punctuation frequency per punctuation symbol."""
    if not text:
        return {}

    punctuation_counts = {symbol: 0 for symbol in string.punctuation}
    total_punctuation = 0
    for char in text:
        if char in punctuation_counts:
            punctuation_counts[char] += 1
            total_punctuation += 1

    if total_punctuation == 0:
        return {}

    frequencies = {
        symbol: safe_div(count, total_punctuation)
        for symbol, count in punctuation_counts.items()
        if count > 0
    }
    return dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))


def shannon_entropy_from_probabilities(probabilities: Iterable[float]) -> float:
    """Compute Shannon entropy from probabilities."""
    entropy = 0.0
    for probability in probabilities:
        probability = max(float(probability), 1e-12)
        entropy += -probability * math.log(probability)
    return float(entropy)


def token_entropy(tokens: Sequence[str]) -> float:
    """Compute token entropy for a sequence of tokens."""
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = float(len(tokens))
    probabilities = [value / total for value in counts.values()]
    return shannon_entropy_from_probabilities(probabilities)


def flatten_feature_dict(feature_map: Dict[str, float], prefix: str) -> Tuple[List[str], List[float]]:
    """Flatten a numeric feature dictionary into parallel name and value lists."""
    names: List[str] = []
    values: List[float] = []
    for key in sorted(feature_map.keys()):
        names.append(f"{prefix}_{key}")
        values.append(float(feature_map[key]))
    return names, values

