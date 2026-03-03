"""Citation detection and citation-aware plagiarism weighting."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from plagiarism_system.utils import split_sentences


CITATION_PATTERNS = {
    "apa": re.compile(r"\([A-Z][A-Za-z\-]+(?:\s+et al\.)?,\s*(19|20)\d{2}[a-z]?\)"),
    "author_year": re.compile(r"\([A-Z][A-Za-z\-]+(?:,\s*[A-Z][A-Za-z\-]+)*,\s*(19|20)\d{2}\)"),
    "numeric_bracket": re.compile(r"\[(\d+([,\-\s]+\d+)*)\]"),
    "ieee_inline": re.compile(r"\b\[\d+\]\s*[A-Z]"),
}


def detect_citations(text: str) -> List[Dict[str, object]]:
    """Detect citations and return matched spans with style labels."""
    findings: List[Dict[str, object]] = []
    for style, pattern in CITATION_PATTERNS.items():
        for match in pattern.finditer(text or ""):
            findings.append(
                {
                    "style": style,
                    "citation": match.group(0),
                    "start": int(match.start()),
                    "end": int(match.end()),
                }
            )
    return sorted(findings, key=lambda row: row["start"])


def sentence_has_citation(sentence: str) -> bool:
    """Return True when a sentence contains a known citation pattern."""
    return bool(detect_citations(sentence))


def citation_aware_adjustment(base_score: float, matched_sentences: Sequence[Dict[str, object]], text: str) -> Dict[str, object]:
    """Reduce plagiarism score weight for matched sentences that include citations."""
    citations = detect_citations(text)
    total_sentences = len(split_sentences(text))
    citation_density = (len(citations) / total_sentences) if total_sentences else 0.0

    adjusted_score = float(base_score)
    adjustments: List[Dict[str, object]] = []

    for item in matched_sentences:
        sentence = str(item.get("target_sentence") or item.get("sentence") or "")
        similarity = float(item.get("similarity", 0.0))
        category = str(item.get("category", "unknown"))
        if not sentence_has_citation(sentence):
            continue

        penalty = min(similarity * 12.0, 8.0)
        adjusted_score = max(0.0, adjusted_score - penalty)
        adjustments.append(
            {
                "sentence": sentence,
                "category": category,
                "similarity": round(similarity, 6),
                "citation_adjustment": round(-penalty, 4),
            }
        )

    adjusted_score = max(0.0, adjusted_score * (1.0 - min(citation_density * 0.15, 0.10)))

    return {
        "citation_count": len(citations),
        "citation_density": round(citation_density, 6),
        "citations_detected": citations,
        "adjustments": adjustments,
        "adjusted_score": round(adjusted_score, 4),
    }

