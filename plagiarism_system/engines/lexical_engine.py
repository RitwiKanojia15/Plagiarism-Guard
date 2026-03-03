"""Lexical plagiarism engine with shingling, MinHash, block matching, and LCS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from plagiarism_system.utils import clean_text, flatten_texts, generate_ngrams, split_sentences, tokenize_words

try:  # pragma: no cover - optional dependency
    from datasketch import MinHash
except Exception:  # pragma: no cover - optional dependency
    MinHash = None


@dataclass(frozen=True)
class LexicalThresholds:
    """Configuration for lexical matching sensitivity."""

    ngram_size: int = 5
    min_exact_block_words: int = 8
    minhash_permutations: int = 128


def _shingles(text: str, n: int = 5) -> Set[str]:
    """Create n-gram shingles from text."""
    tokens = tokenize_words(text)
    return {" ".join(gram) for gram in generate_ngrams(tokens, n)}


def _minhash_signature(shingles: Set[str], num_perm: int = 128):
    """Build MinHash signature when datasketch is available, otherwise return shingles."""
    if MinHash is None:
        return shingles
    signature = MinHash(num_perm=num_perm)
    for item in sorted(shingles):
        signature.update(item.encode("utf-8"))
    return signature


def _signature_similarity(left, right) -> float:
    """Compute similarity between signatures or sets."""
    if MinHash is not None and hasattr(left, "jaccard"):
        return float(left.jaccard(right))
    left_set = set(left or set())
    right_set = set(right or set())
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set) / len(union))


def _lcs_length(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> int:
    """Compute LCS length using dynamic programming."""
    if not tokens_a or not tokens_b:
        return 0

    prev = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        curr = [0]
        for idx, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[idx]))
        prev = curr
    return prev[-1]


def _find_exact_blocks(target_sentences: Sequence[str], source_sentences: Sequence[str], min_words: int) -> List[Dict[str, object]]:
    """Find exact sentence-level copied blocks between target and source sets."""
    source_lookup: Dict[str, List[int]] = {}
    for source_index, source_sentence in enumerate(source_sentences):
        normalized = clean_text(source_sentence).lower()
        if not normalized:
            continue
        source_lookup.setdefault(normalized, []).append(source_index)

    matched_blocks: List[Dict[str, object]] = []
    for target_index, target_sentence in enumerate(target_sentences):
        normalized_target = clean_text(target_sentence).lower()
        if not normalized_target:
            continue
        word_count = len(tokenize_words(normalized_target))
        if word_count < min_words:
            continue
        if normalized_target in source_lookup:
            for source_index in source_lookup[normalized_target]:
                matched_blocks.append(
                    {
                        "target_sentence_index": target_index,
                        "source_sentence_index": source_index,
                        "matched_text": target_sentence,
                        "word_count": word_count,
                    }
                )
    return matched_blocks


def lexical_analysis(target_text: str, source_texts: Sequence[str], thresholds: LexicalThresholds | None = None) -> Dict[str, object]:
    """Run lexical plagiarism analysis and return explainable lexical details."""
    config = thresholds or LexicalThresholds()
    cleaned_target = clean_text(target_text)
    cleaned_sources = [clean_text(source) for source in source_texts if clean_text(source)]

    if not cleaned_target or not cleaned_sources:
        return {
            "exact_similarity_percentage": 0.0,
            "matched_blocks": [],
            "overlapping_ngrams": [],
            "lcs_ratio": 0.0,
            "minhash_similarity": 0.0,
        }

    merged_source = flatten_texts(cleaned_sources)
    target_tokens = tokenize_words(cleaned_target)
    source_tokens = tokenize_words(merged_source)

    target_shingles = _shingles(cleaned_target, config.ngram_size)
    source_shingles = _shingles(merged_source, config.ngram_size)
    target_signature = _minhash_signature(target_shingles, config.minhash_permutations)
    source_signature = _minhash_signature(source_shingles, config.minhash_permutations)
    minhash_similarity = _signature_similarity(target_signature, source_signature)

    overlapping_ngrams = sorted(target_shingles & source_shingles)
    overlap_ratio = len(overlapping_ngrams) / max(len(target_shingles), 1)

    lcs_ratio = _lcs_length(target_tokens, source_tokens) / max(len(target_tokens), 1)

    target_sentences = split_sentences(cleaned_target)
    source_sentences = split_sentences(merged_source)
    matched_blocks = _find_exact_blocks(
        target_sentences=target_sentences,
        source_sentences=source_sentences,
        min_words=config.min_exact_block_words,
    )
    block_ratio = len(matched_blocks) / max(len(target_sentences), 1)

    exact_similarity = (0.45 * minhash_similarity) + (0.25 * overlap_ratio) + (0.20 * lcs_ratio) + (0.10 * block_ratio)
    exact_similarity = max(0.0, min(1.0, exact_similarity))

    return {
        "exact_similarity_percentage": round(exact_similarity * 100.0, 4),
        "matched_blocks": matched_blocks,
        "overlapping_ngrams": overlapping_ngrams[:200],
        "lcs_ratio": round(lcs_ratio, 6),
        "minhash_similarity": round(minhash_similarity, 6),
    }

