"""API and page routes for plagiarism + AI detection platform."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Sequence, Tuple

from flask import Blueprint, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.datastructures import FileStorage

from . import auth
from .database import session_scope
from .models import AnalysisRecord, User
from plagiarism_system.engines import (
    ai_detection_ensemble,
    citation_aware_adjustment,
    explain_ai_prediction,
    lexical_analysis,
    load_or_train_ensemble,
    semantic_analysis,
    stylometric_analysis,
    train_ensemble_models,
)
from plagiarism_system.reports.pdf_export import export_analysis_pdf
from plagiarism_system.utils import clean_text, split_sentences
from plagiarism_system.utils.text_extractor import extract_text

web_blueprint = Blueprint("web", __name__)
SUPPORTED_UPLOAD_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".pdf", ".doc", ".docx"}
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _serialize_user(user: User) -> Dict[str, object]:
    """Serialize user model to API-safe payload."""
    return {
        "id": int(user.id),
        "full_name": str(user.full_name),
        "email": str(user.email),
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _serialize_record(record: AnalysisRecord) -> Dict[str, object]:
    """Serialize analysis record payload."""
    pdf_available = bool(record.notes and os.path.exists(record.notes))
    pdf_download_url = f"/api/reports/{int(record.id)}/pdf" if pdf_available else ""
    return {
        "id": int(record.id),
        "analysis_mode": str(record.analysis_mode),
        "file_name": record.file_name or "",
        "total_similarity": float(record.total_similarity or 0.0),
        "ai_likelihood": float(record.ai_likelihood or 0.0),
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "report": record.report_json or {},
        "pdf_available": pdf_available,
        "pdf_download_url": pdf_download_url,
    }


def _decode_bytes(payload: bytes) -> str:
    """Decode bytes into text with utf-8 fallback sequence."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return payload.decode(encoding)
        except Exception:
            continue
    return payload.decode("utf-8", errors="ignore")


def _extract_uploaded_text(file: FileStorage | None) -> str:
    """Extract plaintext content from uploaded file with document parsers."""
    if file is None:
        return ""
    filename = file.filename or "upload.txt"
    extension = Path(filename).suffix.lower()
    if extension and extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise ValueError(f"Unsupported file type `{extension}`. Allowed types: {supported}")

    content = file.read()
    file.seek(0)
    if not content:
        return ""

    temp_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=extension or ".txt") as handle:
            handle.write(content)
            temp_path = handle.name

        extracted_text, _pages, _meta = extract_text(temp_path, file.content_type or "")
        normalized = clean_text(extracted_text)
        if normalized:
            return normalized

        # Fallback for text-like files if extraction result is empty.
        if extension in {".txt", ".md", ".csv", ".json", ".log"}:
            return clean_text(_decode_bytes(content))

        raise ValueError(f"No readable text found in file `{filename}`.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to parse file `{filename}`: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _parse_source_texts() -> Tuple[str, List[str]]:
    """Parse target text and source texts from JSON or multipart request."""
    text = ""
    source_texts: List[str] = []

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text = clean_text(str(payload.get("text", "")))
        raw_sources = payload.get("source_texts", [])
        if isinstance(raw_sources, list):
            source_texts = [clean_text(str(item)) for item in raw_sources if clean_text(str(item))]
        elif isinstance(raw_sources, str):
            source_texts = [clean_text(raw_sources)] if clean_text(raw_sources) else []
        return text, source_texts

    text = clean_text(request.form.get("text", ""))
    raw_sources = request.form.get("source_texts", "")
    if raw_sources:
        try:
            parsed = json.loads(raw_sources)
            if isinstance(parsed, list):
                source_texts.extend(clean_text(str(item)) for item in parsed if clean_text(str(item)))
            elif isinstance(parsed, str) and clean_text(parsed):
                source_texts.append(clean_text(parsed))
        except json.JSONDecodeError:
            if clean_text(raw_sources):
                source_texts.append(clean_text(raw_sources))

    target_file = request.files.get("document")
    target_file_text = _extract_uploaded_text(target_file)
    if target_file_text:
        text = target_file_text

    for file in request.files.getlist("sources"):
        source_text = _extract_uploaded_text(file)
        if source_text:
            source_texts.append(source_text)

    return text, source_texts


def _parse_analysis_mode() -> str:
    """Parse analysis mode from request payload."""
    raw_mode = "full"
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        raw_mode = str(payload.get("analysis_mode", "full"))
    else:
        raw_mode = str(request.form.get("analysis_mode", "full"))

    mode = raw_mode.strip().lower()
    if mode in {"ai", "ai-only", "ai_only"}:
        return "ai_only"
    if mode not in {"full", "ai_only"}:
        return "full"
    return mode


def _sentence_highlights(
    target_text: str,
    lexical_details: Dict[str, object],
    semantic_details: Dict[str, object],
    ai_probability: float,
) -> List[Dict[str, object]]:
    """Build sentence-level highlighting payload for frontend rendering."""
    sentences = split_sentences(target_text)
    highlights: List[Dict[str, object]] = []

    copied_indexes = {int(item["target_sentence_index"]) for item in lexical_details.get("matched_blocks", [])}
    semantic_lookup = {int(item["target_sentence_index"]): item for item in semantic_details.get("matched_sentences", [])}

    for index, sentence in enumerate(sentences):
        label = "unique"
        color = "transparent"
        reason = "No threshold exceeded."

        semantic_item = semantic_lookup.get(index)
        semantic_category = str(semantic_item.get("category")) if semantic_item else "unique"
        semantic_similarity = float(semantic_item.get("similarity", 0.0)) if semantic_item else 0.0

        if index in copied_indexes or semantic_category == "near_duplicate":
            label = "copied"
            color = "red"
            reason = f"High lexical/semantic overlap (similarity={semantic_similarity:.3f})."
        elif semantic_category == "paraphrased":
            label = "paraphrased"
            color = "yellow"
            reason = f"Semantic paraphrase match (similarity={semantic_similarity:.3f})."
        elif ai_probability >= 0.60:
            label = "ai_likely"
            color = "blue"
            reason = "Global AI-likelihood score is high."

        highlights.append(
            {
                "sentence_index": index,
                "sentence": sentence,
                "label": label,
                "color": color,
                "reason": reason,
            }
        )
    return highlights


def _final_weighted_score(exact_plagiarism: float, paraphrased_content: float, ai_likelihood: float) -> float:
    """Calculate final weighted plagiarism-centric similarity score."""
    weighted = (0.60 * exact_plagiarism) + (0.40 * paraphrased_content)
    weighted = min(100.0, max(0.0, weighted + (0.05 * ai_likelihood)))
    return round(weighted, 4)


def _pipeline_report(target_text: str, source_texts: Sequence[str], analysis_mode: str = "full") -> Dict[str, object]:
    """Run full hybrid pipeline and build structured report JSON."""
    cleaned_target = clean_text(target_text)
    mode = "ai_only" if analysis_mode == "ai_only" else "full"
    cleaned_sources = [clean_text(text) for text in source_texts if clean_text(text)] if mode == "full" else []

    lexical_details = lexical_analysis(cleaned_target, cleaned_sources)
    semantic_details = semantic_analysis(cleaned_target, cleaned_sources)
    stylometry_features = stylometric_analysis(cleaned_target)
    ai_result = ai_detection_ensemble(cleaned_target)

    exact_plagiarism = float(lexical_details.get("exact_similarity_percentage", 0.0)) if mode == "full" else 0.0
    paraphrased_content = float(semantic_details.get("paraphrased_percentage", 0.0)) if mode == "full" else 0.0
    ai_likelihood = float(ai_result.get("ai_probability", 0.0)) * 100.0

    base_total_similarity = _final_weighted_score(exact_plagiarism, paraphrased_content, ai_likelihood) if mode == "full" else 0.0
    citation_adjustment = citation_aware_adjustment(
        base_score=base_total_similarity,
        matched_sentences=semantic_details.get("matched_sentences", []),
        text=cleaned_target,
    )

    total_similarity = float(citation_adjustment.get("adjusted_score", base_total_similarity)) if mode == "full" else 0.0
    confidence = float(ai_result.get("confidence_score", 0.0))

    model_bundle = load_or_train_ensemble()
    ai_explanation = explain_ai_prediction(
        model_bundle=model_bundle,
        feature_vector=ai_result.get("feature_vector", []),
        feature_names=ai_result.get("feature_names", []),
        output_dir=Path(__file__).resolve().parents[1] / "reports",
    )

    highlights = _sentence_highlights(
        target_text=cleaned_target,
        lexical_details=lexical_details,
        semantic_details=semantic_details,
        ai_probability=float(ai_result.get("ai_probability", 0.0)),
    )

    report = {
        "analysis_mode": mode,
        "reference_sources_used": len(cleaned_sources),
        "total_similarity": round(total_similarity, 4),
        "exact_plagiarism": round(exact_plagiarism, 4),
        "paraphrased_content": round(paraphrased_content, 4),
        "self_plagiarism": 0.0,
        "ai_likelihood": round(ai_likelihood, 4),
        "confidence": round(confidence, 4),
        "lexical_details": lexical_details,
        "semantic_details": semantic_details,
        "stylometry_features": stylometry_features,
        "ai_explanation": ai_explanation,
        "ai_detection": ai_result,
        "citation_adjustment": citation_adjustment,
        "sentence_highlighting": highlights,
        "report_version": "1.0.0",
    }
    return report


def _auth_payload() -> Dict[str, str]:
    """Read authentication payload from JSON or form body."""
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form.to_dict() if request.form else {}
    return {key: str(value or "").strip() for key, value in payload.items()}


@web_blueprint.get("/")
def home():
    """Redirect user based on authentication state."""
    if auth.current_user_payload():
        return redirect(url_for("web.dashboard_page"))
    return redirect(url_for("web.login_page"))


@web_blueprint.get("/register")
def register_page():
    """Render registration page."""
    if auth.current_user_payload():
        return redirect(url_for("web.dashboard_page"))
    return render_template("register.html")


@web_blueprint.get("/register.html")
def register_page_html():
    """Backward-compatible register URL alias."""
    return redirect(url_for("web.register_page"))


@web_blueprint.get("/login")
def login_page():
    """Render login page."""
    if auth.current_user_payload():
        return redirect(url_for("web.dashboard_page"))
    return render_template("login.html")


@web_blueprint.get("/login.html")
def login_page_html():
    """Backward-compatible login URL alias."""
    return redirect(url_for("web.login_page"))


@web_blueprint.get("/dashboard")
@auth.login_required
def dashboard_page():
    """Render dashboard page."""
    return render_template("dashboard.html")


@web_blueprint.get("/dashboard.html")
@auth.login_required
def dashboard_page_html():
    """Backward-compatible dashboard URL alias."""
    return redirect(url_for("web.dashboard_page"))


@web_blueprint.get("/upload")
@auth.login_required
def upload_page():
    """Render upload page."""
    return render_template("upload.html")


@web_blueprint.get("/upload.html")
@auth.login_required
def upload_page_html():
    """Backward-compatible upload URL alias."""
    return redirect(url_for("web.upload_page"))


@web_blueprint.get("/result")
@auth.login_required
def result_page():
    """Render result page."""
    return render_template("result.html")


@web_blueprint.get("/result.html")
@auth.login_required
def result_page_html():
    """Backward-compatible result URL alias."""
    return redirect(url_for("web.result_page"))


@web_blueprint.get("/favicon.ico")
def favicon():
    """Return empty favicon response to avoid noisy browser 404s."""
    return "", 204


@web_blueprint.get("/api/health")
def health():
    """Return service health status."""
    return jsonify({"status": "ok", "service": "hybrid-plagiarism-ai-detector"})


@web_blueprint.post("/api/auth/register")
def register_user():
    """Create new user account."""
    payload = _auth_payload()
    full_name = payload.get("full_name", "")
    email = auth.normalize_email(payload.get("email", ""))
    password = payload.get("password", "")

    if len(full_name) < 2:
        return jsonify({"error": "Full name must be at least 2 characters."}), 400
    if not EMAIL_PATTERN.match(email):
        return jsonify({"error": "Valid email is required."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    with session_scope() as db:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            return jsonify({"error": "Email already registered."}), 409

        user = User(
            full_name=full_name,
            email=email,
            password_hash=auth.hash_password(password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return jsonify({"message": "Registration successful.", "user": _serialize_user(user)})


@web_blueprint.post("/api/auth/login")
def login_user():
    """Authenticate user and create session."""
    payload = _auth_payload()
    email = auth.normalize_email(payload.get("email", ""))
    password = payload.get("password", "")

    if not EMAIL_PATTERN.match(email):
        return jsonify({"error": "Valid email is required."}), 400
    if not password:
        return jsonify({"error": "Password is required."}), 400

    with session_scope() as db:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return jsonify({"error": "User not found. Register first."}), 404
        if not auth.verify_password(password, user.password_hash):
            return jsonify({"error": "Invalid credentials."}), 401

        auth.login_user(user)
        return jsonify({"message": "Login successful.", "user": _serialize_user(user)})


@web_blueprint.post("/api/auth/logout")
def logout():
    """Destroy user session."""
    auth.logout_user()
    return jsonify({"message": "Logged out."})


@web_blueprint.get("/api/auth/me")
def auth_me():
    """Get current authenticated user."""
    payload = auth.current_user_payload()
    if not payload:
        return jsonify({"error": "Authentication required"}), 401
    return jsonify({"authenticated": True, "user": payload})


@web_blueprint.get("/api/reports")
@auth.login_required
def list_reports():
    """List latest analysis reports for current user."""
    current = auth.current_user_payload()
    with session_scope() as db:
        rows = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.user_id == int(current["id"]))
            .order_by(AnalysisRecord.created_at.desc())
            .limit(30)
            .all()
        )
        return jsonify(
            {
                "reports": [
                    {
                        "id": int(row.id),
                        "analysis_mode": str(row.analysis_mode),
                        "file_name": row.file_name or "",
                        "total_similarity": float(row.total_similarity or 0.0),
                        "ai_likelihood": float(row.ai_likelihood or 0.0),
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "pdf_available": bool(row.notes and os.path.exists(row.notes)),
                        "pdf_download_url": f"/api/reports/{int(row.id)}/pdf" if (row.notes and os.path.exists(row.notes)) else "",
                    }
                    for row in rows
                ]
            }
        )


@web_blueprint.get("/api/reports/latest")
@auth.login_required
def latest_report():
    """Fetch latest report for current user."""
    current = auth.current_user_payload()
    with session_scope() as db:
        row = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.user_id == int(current["id"]))
            .order_by(AnalysisRecord.created_at.desc())
            .first()
        )
        if not row:
            return jsonify({"error": "No reports available."}), 404
        payload = _serialize_record(row)
        report_payload = payload.get("report") or {}
        if payload.get("pdf_download_url") and isinstance(report_payload, dict):
            report_payload.setdefault("pdf_download_url", payload["pdf_download_url"])
            report_payload.setdefault("record_id", int(row.id))
            payload["report"] = report_payload
        return jsonify(payload)


@web_blueprint.get("/api/reports/<int:record_id>/pdf")
@auth.login_required
def download_report_pdf(record_id: int):
    """Download merged analysis PDF for a given report record."""
    current = auth.current_user_payload()
    with session_scope() as db:
        row = (
            db.query(AnalysisRecord)
            .filter(
                AnalysisRecord.id == int(record_id),
                AnalysisRecord.user_id == int(current["id"]),
            )
            .first()
        )
        if not row:
            return jsonify({"error": "Report not found."}), 404
        if not row.notes or not os.path.exists(row.notes):
            return jsonify({"error": "Analysis PDF not available for this report."}), 404

        download_name = f"analysis_report_{int(row.id)}.pdf"
        return send_file(
            row.notes,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=download_name,
        )


@web_blueprint.post("/api/train-ai-model")
@auth.login_required
def train_ai_model():
    """Train AI ensemble model and persist artifacts."""
    payload = request.get_json(silent=True) or {}
    sample_count = int(payload.get("sample_count", 300))
    random_seed = int(payload.get("random_seed", 42))
    result = train_ensemble_models(sample_count=sample_count, random_seed=random_seed)
    return jsonify(result)


@web_blueprint.post("/api/analyze")
@auth.login_required
def analyze():
    """Run full plagiarism + AI analysis pipeline."""
    try:
        target_text, source_texts = _parse_source_texts()
        analysis_mode = _parse_analysis_mode()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not target_text:
        return jsonify({"error": "No target text provided. Submit `text` or `document`."}), 400
    if analysis_mode == "full" and not source_texts:
        return jsonify({"error": "No source texts provided. Submit `source_texts` or `sources`."}), 400

    try:
        report = _pipeline_report(target_text=target_text, source_texts=source_texts, analysis_mode=analysis_mode)
        current = auth.current_user_payload()
        if current:
            file_name = "pasted_text.txt"
            uploaded = request.files.get("document")
            if uploaded and uploaded.filename:
                file_name = uploaded.filename
            with session_scope() as db:
                record = AnalysisRecord(
                    user_id=int(current["id"]),
                    file_name=file_name,
                    analysis_mode=analysis_mode,
                    total_similarity=float(report.get("total_similarity", 0.0)),
                    ai_likelihood=float(report.get("ai_likelihood", 0.0)),
                    report_json=report,
                )
                db.add(record)
                db.commit()
                db.refresh(record)

                pdf_path, pdf_error = export_analysis_pdf(
                    uploaded_file=uploaded,
                    report=report,
                    reports_dir=Path(__file__).resolve().parents[1] / "reports",
                    file_tag=f"user_{int(current['id'])}_record_{int(record.id)}",
                    fallback_text=target_text,
                )
                if pdf_path:
                    record.notes = pdf_path
                    report["pdf_download_url"] = f"/api/reports/{int(record.id)}/pdf"
                elif pdf_error:
                    report["pdf_generation_warning"] = str(pdf_error)

                report["record_id"] = int(record.id)
                report["created_at"] = record.created_at.isoformat() if record.created_at else None
                record.report_json = report
                db.add(record)
                db.commit()
        return jsonify(report)
    except Exception as exc:  # pragma: no cover - runtime safety
        return jsonify({"error": "Analysis failed", "detail": str(exc)}), 500
