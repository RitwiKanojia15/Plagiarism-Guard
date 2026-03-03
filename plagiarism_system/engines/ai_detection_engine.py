"""AI detection ensemble engine combining language, stylometric, and structural signals."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plagiarism_system.engines.stylometry_engine import stylometric_analysis
from plagiarism_system.utils import clean_text, lexical_diversity, split_sentences, token_entropy, tokenize_words

_TORCH_DISABLED = os.getenv("PLAGIARISM_DISABLE_TORCH", "").strip().lower() in {"1", "true", "yes", "on"}
if _TORCH_DISABLED:  # pragma: no cover - optional dependency
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
else:
    try:  # pragma: no cover - optional dependency
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except BaseException:  # pragma: no cover - optional dependency
        torch = None
        AutoModelForCausalLM = None
        AutoTokenizer = None


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ai_classifier.pkl"
MAX_MODEL_TOKENS = 512
EPSILON = 1e-12

_MODEL_CACHE: Dict[str, object] = {"bundle": None, "path": None}
_LM_CACHE: Dict[str, object] = {"tokenizer": None, "model": None, "backend": "heuristic"}


AI_TEMPLATES = [
    "This section provides a structured overview of {topic} and presents consistent observations for broader understanding.",
    "The analysis demonstrates key aspects of {topic} through a clear and systematic explanation.",
    "In addition, the discussion highlights practical implications of {topic} for contemporary applications.",
    "Overall, this synthesis reinforces the importance of {topic} and identifies future areas of inquiry.",
]

HUMAN_TEMPLATES = [
    "I first thought the result looked obvious, then an outlier forced me to recalculate everything.",
    "During revision, one paragraph felt robotic, so I replaced generic wording with specific examples from our tests.",
    "The method worked in simulation, but real inputs exposed edge cases we had not anticipated.",
    "After comparing drafts, I kept the shorter sentence and rewrote the longer one to explain my actual reasoning.",
]

POS_KEYS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ", "NUM", "PRT", ".", "X"]
PUNCT_KEYS = [".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "-", "'", "\""]


def _resolve_device() -> str:
    """Return best available torch device."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_language_model() -> Tuple[object, object, str]:
    """Load GPT-2 family model for perplexity and token probability estimates."""
    if _LM_CACHE["tokenizer"] is not None and _LM_CACHE["model"] is not None:
        return _LM_CACHE["tokenizer"], _LM_CACHE["model"], str(_LM_CACHE["backend"])

    if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
        _LM_CACHE["backend"] = "heuristic"
        return None, None, "heuristic"

    for model_name in ("distilgpt2", "gpt2"):
        for local_only in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
                model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_only)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                model.eval()
                model.to(_resolve_device())
                backend = f"transformers:{model_name}:{'local' if local_only else 'downloaded'}"
                _LM_CACHE["tokenizer"] = tokenizer
                _LM_CACHE["model"] = model
                _LM_CACHE["backend"] = backend
                return tokenizer, model, backend
            except Exception:
                continue

    _LM_CACHE["backend"] = "heuristic"
    return None, None, "heuristic"


def _heuristic_perplexity(text: str) -> float:
    """Compute smoothed unigram perplexity as fallback."""
    tokens = tokenize_words(text)
    if len(tokens) < 2:
        return 220.0
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = float(len(tokens))
    vocab = float(len(counts))
    neg_log = 0.0
    for token in tokens:
        probability = (counts[token] + 1.0) / (total + vocab)
        neg_log += -math.log(max(probability, EPSILON))
    cross_entropy = neg_log / total
    return float(min(max(math.exp(min(cross_entropy, 50.0)), 1.0), 1000.0))


def perplexity_score(text: str) -> float:
    """Compute perplexity from GPT-2 model with fallback."""
    tokenizer, model, _backend = _load_language_model()
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    if tokenizer is None or model is None or torch is None:
        return round(_heuristic_perplexity(cleaned), 6)
    try:
        device = next(model.parameters()).device
        encoded = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_MODEL_TOKENS)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded, labels=encoded["input_ids"])
        loss = float(output.loss.item())
        perplexity = math.exp(min(loss, 20.0))
        return round(float(perplexity), 6)
    except Exception:
        return round(_heuristic_perplexity(cleaned), 6)


def burstiness_score(sentences: Sequence[str]) -> float:
    """Compute burstiness score based on sentence-length variability."""
    lengths = [len(tokenize_words(sentence)) for sentence in sentences if sentence.strip()]
    lengths = [length for length in lengths if length > 0]
    if len(lengths) <= 1:
        return 0.0
    values = np.array(lengths, dtype=float)
    std = float(values.std())
    mean = float(values.mean())
    coefficient = std / max(mean, EPSILON)
    return round(coefficient, 6)


def _synthetic_corpus(sample_count: int, random_seed: int) -> Tuple[List[str], np.ndarray]:
    """Build synthetic AI/human corpus for model bootstrap training."""
    rng = random.Random(random_seed)
    count = max(40, int(sample_count))
    half = count // 2
    texts: List[str] = []
    labels: List[int] = []

    topics = ["machine learning", "climate policy", "cybersecurity", "education technology", "health analytics"]
    for _ in range(half):
        topic = rng.choice(topics)
        length = rng.randint(4, 6)
        text = " ".join(rng.choice(AI_TEMPLATES).format(topic=topic) for _ in range(length))
        texts.append(text)
        labels.append(1)

    for _ in range(half):
        length = rng.randint(3, 6)
        text = " ".join(rng.choice(HUMAN_TEMPLATES) for _ in range(length))
        texts.append(text)
        labels.append(0)

    items = list(zip(texts, labels))
    rng.shuffle(items)
    shuffled_texts, shuffled_labels = zip(*items)
    return list(shuffled_texts), np.array(shuffled_labels, dtype=int)


def _extract_feature_vector(text: str) -> Tuple[np.ndarray, List[str], Dict[str, float], Dict[str, object]]:
    """Extract AI detection feature vector from hybrid linguistic signals."""
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    tokens = tokenize_words(cleaned)

    perplexity = perplexity_score(cleaned)
    burstiness = burstiness_score(sentences)
    entropy = round(token_entropy(tokens), 6)
    lexical_div = round(lexical_diversity(tokens), 6)

    style = stylometric_analysis(cleaned)
    avg_sentence_len = float(style.get("avg_sentence_length", 0.0))
    pos_distribution = style.get("pos_tag_distribution", {}) or {}
    punctuation_distribution = style.get("punctuation_pattern_frequency", {}) or {}

    style_names = [
        "type_token_ratio",
        "hapax_legomena_ratio",
        "sentence_length_variance",
        "function_word_ratio",
        "passive_voice_ratio",
    ] + [f"pos_{key}" for key in POS_KEYS] + [f"punct_{key}" for key in PUNCT_KEYS]

    style_vector = [
        float(style.get("type_token_ratio", 0.0)),
        float(style.get("hapax_legomena_ratio", 0.0)),
        float(style.get("sentence_length_variance", 0.0)),
        float(style.get("function_word_ratio", 0.0)),
        float(style.get("passive_voice_ratio", 0.0)),
    ]
    style_vector.extend(float(pos_distribution.get(key, 0.0)) for key in POS_KEYS)
    style_vector.extend(float(punctuation_distribution.get(key, 0.0)) for key in PUNCT_KEYS)

    feature_names = [
        "perplexity",
        "burstiness",
        "entropy",
        "lexical_diversity",
        "avg_sentence_length",
    ] + style_names
    feature_values = [perplexity, burstiness, entropy, lexical_div, avg_sentence_len] + style_vector
    vector = np.array(feature_values, dtype=float)

    summary_features = {
        "perplexity": perplexity,
        "burstiness": burstiness,
        "entropy": entropy,
        "lexical_diversity": lexical_div,
        "avg_sentence_length": round(avg_sentence_len, 6),
    }
    return vector, feature_names, summary_features, style


def train_ensemble_models(model_path: str | Path = MODEL_PATH, sample_count: int = 300, random_seed: int = 42) -> Dict[str, object]:
    """Train and persist logistic-regression + random-forest ensemble."""
    resolved = Path(model_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    texts, labels = _synthetic_corpus(sample_count=sample_count, random_seed=random_seed)
    extracted = [_extract_feature_vector(text) for text in texts]
    features = np.array([row[0] for row in extracted], dtype=float)
    feature_names = extracted[0][1] if extracted else []

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_seed,
        stratify=labels,
    )

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=random_seed)),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=random_seed,
        n_jobs=-1,
    )

    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    pred_lr = logistic.predict(x_test)
    pred_rf = forest.predict(x_test)
    pred_ens = (0.5 * logistic.predict_proba(x_test)[:, 1]) + (0.5 * forest.predict_proba(x_test)[:, 1])
    pred_ens_label = (pred_ens >= 0.5).astype(int)

    def _metrics(y_true, y_pred) -> Dict[str, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1_score": round(float(f1), 6),
        }

    metrics = {
        "logistic_regression": _metrics(y_test, pred_lr),
        "random_forest": _metrics(y_test, pred_rf),
        "ensemble": _metrics(y_test, pred_ens_label),
    }

    bundle = {
        "feature_names": feature_names,
        "logistic_regression": logistic,
        "random_forest": forest,
        "metrics": metrics,
        "random_seed": random_seed,
        "sample_count": int(len(labels)),
    }
    joblib.dump(bundle, resolved)
    _MODEL_CACHE["bundle"] = bundle
    _MODEL_CACHE["path"] = str(resolved)

    return {
        "model_path": str(resolved),
        "sample_count": int(len(labels)),
        "metrics": metrics,
    }


def load_or_train_ensemble(model_path: str | Path = MODEL_PATH, auto_train: bool = True) -> Dict[str, object]:
    """Load persisted ensemble model bundle or train if missing."""
    resolved = Path(model_path).resolve()
    if _MODEL_CACHE["bundle"] is not None and _MODEL_CACHE["path"] == str(resolved):
        return _MODEL_CACHE["bundle"]

    if not resolved.exists():
        if not auto_train:
            raise FileNotFoundError(f"AI model file not found: {resolved}")
        train_ensemble_models(model_path=resolved)

    bundle = joblib.load(resolved)
    _MODEL_CACHE["bundle"] = bundle
    _MODEL_CACHE["path"] = str(resolved)
    return bundle


def ai_detection_ensemble(text: str, model_path: str | Path = MODEL_PATH, auto_train: bool = True) -> Dict[str, object]:
    """Predict AI probability using logistic-regression and random-forest ensemble."""
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "ai_probability": 0.0,
            "human_probability": 1.0,
            "confidence_score": 0.0,
            "model_breakdown": {"logistic_regression": 0.0, "random_forest": 0.0},
            "feature_snapshot": {},
            "feature_vector": [],
            "feature_names": [],
            "model_path": str(Path(model_path).resolve()),
            "model_metrics": {},
        }

    bundle = load_or_train_ensemble(model_path=model_path, auto_train=auto_train)
    vector, feature_names, summary_features, style = _extract_feature_vector(cleaned)
    vector_2d = vector.reshape(1, -1)

    logistic = bundle["logistic_regression"]
    forest = bundle["random_forest"]
    lr_probability = float(logistic.predict_proba(vector_2d)[0][1])
    rf_probability = float(forest.predict_proba(vector_2d)[0][1])
    ai_probability = (0.5 * lr_probability) + (0.5 * rf_probability)
    human_probability = 1.0 - ai_probability
    confidence_score = min(1.0, abs(ai_probability - 0.5) * 2.0)

    return {
        "ai_probability": round(ai_probability, 6),
        "human_probability": round(human_probability, 6),
        "confidence_score": round(confidence_score, 6),
        "model_breakdown": {
            "logistic_regression": round(lr_probability, 6),
            "random_forest": round(rf_probability, 6),
        },
        "feature_snapshot": summary_features,
        "feature_vector": [round(float(value), 6) for value in vector.tolist()],
        "feature_names": feature_names,
        "stylometry_features": style,
        "model_path": str(Path(model_path).resolve()),
        "model_metrics": bundle.get("metrics", {}),
    }
