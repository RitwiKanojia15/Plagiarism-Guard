"""Semantic paraphrasing engine using sentence embeddings and cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from plagiarism_system.utils import clean_text, split_sentences

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


_EMBEDDING_CACHE: Dict[str, object] = {"model": None, "backend": "tfidf-fallback"}


@dataclass(frozen=True)
class SemanticThresholds:
    """Threshold configuration for semantic similarity classification."""

    near_duplicate_threshold: float = 0.90
    paraphrase_threshold: float = 0.75


def _embedding_model():
    """Load and cache sentence transformer model with fallback behavior."""
    if _EMBEDDING_CACHE["model"] is not None:
        return _EMBEDDING_CACHE["model"], _EMBEDDING_CACHE["backend"]

    if SentenceTransformer is None:
        _EMBEDDING_CACHE["backend"] = "tfidf-fallback"
        return None, "tfidf-fallback"

    for local_only in (True, False):
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=local_only)
            backend = f"sentence-transformers:{'local' if local_only else 'downloaded'}"
            _EMBEDDING_CACHE["model"] = model
            _EMBEDDING_CACHE["backend"] = backend
            return model, backend
        except Exception:
            continue

    _EMBEDDING_CACHE["backend"] = "tfidf-fallback"
    return None, "tfidf-fallback"


def _encode_sentences(sentences: Sequence[str]) -> np.ndarray:
    """Encode sentences into vector space using SBERT or TF-IDF fallback."""
    model, _backend = _embedding_model()
    if not sentences:
        return np.zeros((0, 1), dtype="float32")

    if model is not None:
        embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.astype("float32")

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4096)
    matrix = vectorizer.fit_transform(sentences)
    dense = matrix.toarray().astype("float32")
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


def _cluster_matches(matches: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Group contiguous target sentence matches into cluster blocks."""
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda row: int(row["target_sentence_index"]))
    clusters: List[Dict[str, object]] = []
    current_cluster: Dict[str, object] | None = None

    for match in sorted_matches:
        target_idx = int(match["target_sentence_index"])
        category = str(match["category"])
        similarity = float(match["similarity"])

        if current_cluster is None:
            current_cluster = {
                "start_target_sentence_index": target_idx,
                "end_target_sentence_index": target_idx,
                "category": category,
                "mean_similarity": similarity,
                "sentence_count": 1,
            }
            continue

        contiguous = target_idx == int(current_cluster["end_target_sentence_index"]) + 1
        same_category = category == current_cluster["category"]
        if contiguous and same_category:
            count = int(current_cluster["sentence_count"]) + 1
            current_cluster["end_target_sentence_index"] = target_idx
            current_cluster["sentence_count"] = count
            current_cluster["mean_similarity"] = (
                (float(current_cluster["mean_similarity"]) * (count - 1) + similarity) / count
            )
        else:
            clusters.append(current_cluster)
            current_cluster = {
                "start_target_sentence_index": target_idx,
                "end_target_sentence_index": target_idx,
                "category": category,
                "mean_similarity": similarity,
                "sentence_count": 1,
            }

    if current_cluster is not None:
        clusters.append(current_cluster)

    for cluster in clusters:
        cluster["mean_similarity"] = round(float(cluster["mean_similarity"]), 6)

    return clusters


def semantic_analysis(
    target_text: str,
    source_texts: Sequence[str],
    thresholds: SemanticThresholds | None = None,
) -> Dict[str, object]:
    """Detect paraphrased and near-duplicate semantic overlaps."""
    config = thresholds or SemanticThresholds()
    target_sentences = split_sentences(clean_text(target_text))

    source_sentence_rows: List[Dict[str, object]] = []
    for source_index, source_text in enumerate(source_texts):
        for sentence in split_sentences(clean_text(source_text)):
            source_sentence_rows.append(
                {
                    "source_text_index": source_index,
                    "sentence": sentence,
                }
            )

    if not target_sentences or not source_sentence_rows:
        return {
            "paraphrased_percentage": 0.0,
            "near_duplicate_percentage": 0.0,
            "similarity_matrix": [],
            "matched_sentences": [],
            "clusters": [],
            "embedding_backend": _EMBEDDING_CACHE["backend"],
        }

    source_sentences = [row["sentence"] for row in source_sentence_rows]
    target_embeddings = _encode_sentences(target_sentences)
    source_embeddings = _encode_sentences(source_sentences)
    similarity_matrix = cosine_similarity(target_embeddings, source_embeddings)

    matched_sentences: List[Dict[str, object]] = []
    paraphrased_count = 0
    near_duplicate_count = 0

    for target_idx, target_sentence in enumerate(target_sentences):
        row = similarity_matrix[target_idx]
        best_index = int(np.argmax(row))
        best_similarity = float(row[best_index])
        source_row = source_sentence_rows[best_index]
        category = "unique"
        if best_similarity >= config.near_duplicate_threshold:
            category = "near_duplicate"
            near_duplicate_count += 1
        elif best_similarity >= config.paraphrase_threshold:
            category = "paraphrased"
            paraphrased_count += 1

        matched_sentences.append(
            {
                "target_sentence_index": target_idx,
                "target_sentence": target_sentence,
                "source_sentence": source_row["sentence"],
                "source_text_index": int(source_row["source_text_index"]),
                "similarity": round(best_similarity, 6),
                "category": category,
            }
        )

    flagged_matches = [row for row in matched_sentences if row["category"] in {"near_duplicate", "paraphrased"}]
    clusters = _cluster_matches(flagged_matches)

    total_target = max(len(target_sentences), 1)
    paraphrased_percentage = ((paraphrased_count + near_duplicate_count) / total_target) * 100.0
    near_duplicate_percentage = (near_duplicate_count / total_target) * 100.0

    rounded_matrix = [[round(float(value), 6) for value in row] for row in similarity_matrix.tolist()]

    return {
        "paraphrased_percentage": round(paraphrased_percentage, 4),
        "near_duplicate_percentage": round(near_duplicate_percentage, 4),
        "similarity_matrix": rounded_matrix,
        "matched_sentences": matched_sentences,
        "clusters": clusters,
        "embedding_backend": _EMBEDDING_CACHE["backend"],
    }