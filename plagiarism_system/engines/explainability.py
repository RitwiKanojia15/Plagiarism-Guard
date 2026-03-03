"""Explainability module for AI detection predictions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def _top_feature_rows(feature_names: Sequence[str], values: Sequence[float], top_k: int = 12) -> List[Dict[str, float]]:
    """Return top absolute-valued feature rows for explanation."""
    rows = []
    for name, value in zip(feature_names, values):
        rows.append({"feature": str(name), "impact": float(value), "abs_impact": abs(float(value))})
    rows.sort(key=lambda row: row["abs_impact"], reverse=True)
    return rows[:top_k]


def _save_waterfall_plot(rows: Sequence[Dict[str, float]], output_dir: Path) -> str | None:
    """Save a simple waterfall-like horizontal feature impact chart."""
    if plt is None or not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"waterfall_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png"

    features = [row["feature"] for row in rows]
    impacts = [row["impact"] for row in rows]
    colors = ["#247ba0" if value >= 0 else "#d62828" for value in impacts]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(features[::-1], impacts[::-1], color=colors[::-1])
    ax.axvline(0.0, color="#111111", linewidth=1.2)
    ax.set_xlabel("Feature Impact")
    ax.set_ylabel("Feature")
    ax.set_title("AI Detection Feature Contribution (Waterfall View)")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def explain_ai_prediction(
    model_bundle: Dict[str, object],
    feature_vector: Sequence[float],
    feature_names: Sequence[str],
    output_dir: str | Path | None = None,
) -> Dict[str, object]:
    """Generate feature-importance explanation using SHAP when available."""
    vector = np.array(feature_vector, dtype=float).reshape(1, -1)
    resolved_output = Path(output_dir).resolve() if output_dir else (Path(__file__).resolve().parents[1] / "reports")

    if shap is not None:
        try:
            forest = model_bundle["random_forest"]
            explainer = shap.TreeExplainer(forest)
            shap_values = explainer.shap_values(vector)
            if isinstance(shap_values, list):
                contributions = np.array(shap_values[-1]).reshape(-1)
            else:
                contributions = np.array(shap_values).reshape(-1)
            rows = _top_feature_rows(feature_names, contributions.tolist(), top_k=15)
            path = _save_waterfall_plot(rows, resolved_output)
            return {
                "backend": "shap",
                "top_contributing_features": [
                    {
                        "feature": row["feature"],
                        "impact": round(float(row["impact"]), 6),
                        "abs_impact": round(float(row["abs_impact"]), 6),
                    }
                    for row in rows
                ],
                "waterfall_plot_path": path,
            }
        except Exception:
            pass

    forest = model_bundle.get("random_forest")
    importances = list(getattr(forest, "feature_importances_", [])) if forest is not None else []
    if len(importances) != len(feature_names):
        importances = [0.0 for _ in feature_names]
    rows = _top_feature_rows(feature_names, importances, top_k=15)
    path = _save_waterfall_plot(rows, resolved_output)
    return {
        "backend": "feature_importance_fallback",
        "top_contributing_features": [
            {
                "feature": row["feature"],
                "impact": round(float(row["impact"]), 6),
                "abs_impact": round(float(row["abs_impact"]), 6),
            }
            for row in rows
        ],
        "waterfall_plot_path": path,
    }

