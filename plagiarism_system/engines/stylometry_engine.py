"""Stylometric analysis engine for author-style and AI-pattern features."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from plagiarism_system.utils import (
    clean_text,
    flatten_feature_dict,
    hapax_legomena_ratio,
    lexical_diversity,
    punctuation_pattern_frequency,
    safe_div,
    sentence_length_statistics,
    split_sentences,
    tokenize_words,
)

try:  # pragma: no cover - optional dependency
    import spacy
except Exception:  # pragma: no cover - optional dependency
    spacy = None

try:  # pragma: no cover - optional dependency
    import nltk
except Exception:  # pragma: no cover - optional dependency
    nltk = None


FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while", "to", "of", "in",
    "on", "for", "with", "as", "at", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "do", "does", "did",
    "have", "has", "had", "can", "could", "would", "should", "will", "may", "might", "must",
}

PASSIVE_PATTERN = re.compile(
    r"\b(am|is|are|was|were|be|been|being)\s+\w+(ed|en)\b",
    re.IGNORECASE,
)

POS_TAG_ORDER = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ", "NUM", "PRT", ".", "X"]

_POS_STATE: Dict[str, object] = {"tagger": None, "backend": "heuristic"}


def _spacy_tagger():
    """Load spaCy model for POS tagging."""
    if _POS_STATE["tagger"] is not None:
        return _POS_STATE["tagger"], _POS_STATE["backend"]
    if spacy is None:
        return None, "heuristic"

    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
            _POS_STATE["tagger"] = nlp
            _POS_STATE["backend"] = f"spacy:{model_name}"
            return nlp, _POS_STATE["backend"]
        except Exception:
            continue
    return None, "heuristic"


def _heuristic_pos_tag(token: str) -> str:
    """Assign a coarse POS tag with simple suffix and lexicon heuristics."""
    if token.isdigit():
        return "NUM"
    if token in {"and", "or", "but"}:
        return "CONJ"
    if token in {"in", "on", "at", "to", "from", "for", "with", "of"}:
        return "ADP"
    if token in {"a", "an", "the", "this", "that", "these", "those"}:
        return "DET"
    if token in {"i", "you", "he", "she", "we", "they", "it"}:
        return "PRON"
    if token.endswith("ly"):
        return "ADV"
    if token.endswith(("ed", "ing", "ify", "ise", "ize")):
        return "VERB"
    if token.endswith(("ous", "ive", "able", "al", "ful", "ic")):
        return "ADJ"
    return "NOUN"


def _pos_distribution(tokens: Sequence[str]) -> Tuple[Dict[str, float], str]:
    """Compute normalized POS tag distribution with fallback strategy."""
    if not tokens:
        return {tag: 0.0 for tag in POS_TAG_ORDER}, "heuristic"

    counts = {tag: 0 for tag in POS_TAG_ORDER}

    tagger, backend = _spacy_tagger()
    if tagger is not None:
        doc = tagger(" ".join(tokens))
        for token in doc:
            tag = token.pos_ if token.pos_ in counts else "X"
            counts[tag] += 1
        total = max(sum(counts.values()), 1)
        return {tag: round(safe_div(value, total), 6) for tag, value in counts.items()}, backend

    if nltk is not None:
        try:
            tagged = nltk.pos_tag(list(tokens), tagset="universal")
            mapping = {
                "NOUN": "NOUN",
                "VERB": "VERB",
                "ADJ": "ADJ",
                "ADV": "ADV",
                "PRON": "PRON",
                "DET": "DET",
                "ADP": "ADP",
                "CONJ": "CONJ",
                "NUM": "NUM",
                "PRT": "PRT",
                ".": ".",
                "X": "X",
            }
            for _word, tag in tagged:
                counts[mapping.get(tag, "X")] += 1
            total = max(sum(counts.values()), 1)
            return {tag: round(safe_div(value, total), 6) for tag, value in counts.items()}, "nltk:universal"
        except Exception:
            pass

    for token in tokens:
        counts[_heuristic_pos_tag(token)] += 1
    total = max(sum(counts.values()), 1)
    return {tag: round(safe_div(value, total), 6) for tag, value in counts.items()}, "heuristic"


def _function_word_frequency(tokens: Sequence[str]) -> Dict[str, float]:
    """Compute normalized function-word frequencies for frequent function words."""
    if not tokens:
        return {}
    total = len(tokens)
    frequencies: Dict[str, float] = {}
    for word in FUNCTION_WORDS:
        count = tokens.count(word)
        if count > 0:
            frequencies[word] = round(safe_div(count, total), 6)
    return dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True)[:20])


def _passive_voice_ratio(sentences: Sequence[str]) -> float:
    """Estimate passive voice ratio with pattern-based detection."""
    if not sentences:
        return 0.0
    passive_count = sum(1 for sentence in sentences if PASSIVE_PATTERN.search(sentence))
    return round(safe_div(passive_count, len(sentences)), 6)


def stylometric_analysis(text: str) -> Dict[str, object]:
    """Extract stylometric feature vector and granular linguistic metrics."""
    cleaned = clean_text(text)
    tokens = tokenize_words(cleaned)
    sentences = split_sentences(cleaned)

    ttr = lexical_diversity(tokens)
    hapax_ratio = hapax_legomena_ratio(tokens)
    sentence_stats = sentence_length_statistics(sentences, tokenize_words)
    function_word_freq = _function_word_frequency(tokens)
    function_word_ratio = round(sum(function_word_freq.values()), 6)
    passive_ratio = _passive_voice_ratio(sentences)
    pos_distribution, pos_backend = _pos_distribution(tokens)
    punctuation_freq = punctuation_pattern_frequency(cleaned)

    vector_feature_names = [
        "type_token_ratio",
        "hapax_legomena_ratio",
        "avg_sentence_length",
        "sentence_length_variance",
        "function_word_ratio",
        "passive_voice_ratio",
    ]
    vector_values = [
        round(ttr, 6),
        round(hapax_ratio, 6),
        round(sentence_stats["avg_sentence_length"], 6),
        round(sentence_stats["sentence_length_variance"], 6),
        function_word_ratio,
        passive_ratio,
    ]

    pos_names, pos_values = flatten_feature_dict(pos_distribution, "pos")
    punct_names, punct_values = flatten_feature_dict(punctuation_freq, "punct")
    vector_feature_names.extend(pos_names)
    vector_feature_names.extend(punct_names)
    vector_values.extend([round(value, 6) for value in pos_values + punct_values])

    return {
        "type_token_ratio": round(ttr, 6),
        "hapax_legomena_ratio": round(hapax_ratio, 6),
        "avg_sentence_length": round(sentence_stats["avg_sentence_length"], 6),
        "sentence_length_variance": round(sentence_stats["sentence_length_variance"], 6),
        "function_word_frequency": function_word_freq,
        "function_word_ratio": function_word_ratio,
        "passive_voice_ratio": passive_ratio,
        "pos_tag_distribution": pos_distribution,
        "punctuation_pattern_frequency": punctuation_freq,
        "stylometric_vector": vector_values,
        "stylometric_vector_features": vector_feature_names,
        "pos_backend": pos_backend,
    }

