"""PDF export helpers for merged source-document + analysis summary reports."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from werkzeug.datastructures import FileStorage

from plagiarism_system.utils import tokenize_words
from plagiarism_system.utils.text_extractor import extract_text

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}
DOC_EXTENSIONS = {".docx", ".doc"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
HIGHLIGHT_COLOR = {
    "copied": (1.0, 0.35, 0.35),
    "paraphrased": (1.0, 0.85, 0.35),
    "ai_likely": (0.45, 0.65, 1.0),
    "unique": (0.72, 0.9, 0.72),
}
HIGHLIGHT_ALPHA = {
    "copied": 0.25,
    "paraphrased": 0.25,
    "ai_likely": 0.25,
    "unique": 0.12,
}

try:  # pragma: no cover - optional dependency
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None


def _safe_name(value: str) -> str:
    """Create filesystem-safe slug-like fragment."""
    sanitized = "".join(char if char.isalnum() else "_" for char in value)
    return sanitized.strip("_") or "report"


def _normalize_token(value: str) -> str:
    """Normalize token for text matching."""
    match = TOKEN_PATTERN.search((value or "").lower())
    return match.group(0) if match else ""


def _sentence_candidates(report: Dict[str, object]) -> List[Tuple[List[str], str]]:
    """Collect tokenized sentences to highlight from report labels."""
    rows = report.get("sentence_highlighting", []) or []
    candidates: List[Tuple[List[str], str]] = []
    for row in rows:
        label = str(row.get("label", ""))
        if label not in HIGHLIGHT_COLOR:
            continue
        sentence = str(row.get("sentence", "")).strip()
        tokens = [_normalize_token(token) for token in tokenize_words(sentence)]
        tokens = [token for token in tokens if token]
        if len(tokens) >= 4:
            candidates.append((tokens, label))
    return candidates


def _find_token_spans(page_tokens: Sequence[str], query_tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Find matching token spans for a query sentence in page token stream."""
    if not page_tokens or not query_tokens:
        return []

    spans: List[Tuple[int, int]] = []
    q_len = len(query_tokens)

    # 1) exact full-sentence contiguous search
    if q_len <= len(page_tokens):
        for idx in range(0, len(page_tokens) - q_len + 1):
            if list(page_tokens[idx : idx + q_len]) == list(query_tokens):
                spans.append((idx, idx + q_len - 1))

    if spans:
        return spans

    # 2) fallback partial prefix matching for long sentences
    for size in (12, 10, 8, 6, 5, 4):
        if q_len < size:
            continue
        prefix = list(query_tokens[:size])
        for idx in range(0, len(page_tokens) - size + 1):
            if list(page_tokens[idx : idx + size]) == prefix:
                spans.append((idx, idx + size - 1))
        if spans:
            return spans

    return spans


def _build_highlight_rectangles(source_pdf: Path, report: Dict[str, object]) -> List[Dict[str, object]]:
    """Build highlight rectangles by matching analyzed sentences to PDF word coordinates."""
    if pdfplumber is None:
        return []

    candidates = _sentence_candidates(report)
    if not candidates:
        return []

    rectangles: List[Dict[str, object]] = []
    max_rectangles = 1200

    with pdfplumber.open(str(source_pdf)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True,
            ) or []
            if not words:
                continue

            page_tokens = [_normalize_token(str(word.get("text", ""))) for word in words]
            for query_tokens, label in candidates:
                spans = _find_token_spans(page_tokens, query_tokens)
                if not spans:
                    continue

                for start, end in spans[:4]:
                    for idx in range(start, end + 1):
                        word = words[idx]
                        x0 = float(word.get("x0", 0.0))
                        x1 = float(word.get("x1", 0.0))
                        top = float(word.get("top", 0.0))
                        bottom = float(word.get("bottom", top))
                        rect = {
                            "page_index": page_index,
                            "x0": x0,
                            "x1": x1,
                            # Convert from pdfplumber top-origin to reportlab bottom-origin.
                            "y0": float(page.height) - bottom,
                            "y1": float(page.height) - top,
                            "label": label,
                        }
                        rectangles.append(rect)
                        if len(rectangles) >= max_rectangles:
                            return rectangles
    return rectangles


def _apply_highlight_overlay(source_pdf: Path, report: Dict[str, object], output_pdf: Path) -> bool:
    """Apply sentence-level highlight overlay on source PDF while preserving original structure."""
    rectangles = _build_highlight_rectangles(source_pdf, report)
    if not rectangles:
        return False

    grouped: Dict[int, List[Dict[str, object]]] = {}
    for rect in rectangles:
        grouped.setdefault(int(rect["page_index"]), []).append(rect)

    overlay_pdf = output_pdf.with_suffix(".overlay.pdf")
    source_reader = PdfReader(str(source_pdf))

    overlay_canvas = canvas.Canvas(str(overlay_pdf))
    for page_index, page in enumerate(source_reader.pages):
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)
        overlay_canvas.setPageSize((page_width, page_height))

        for rect in grouped.get(page_index, []):
            label = str(rect.get("label", "copied"))
            color = HIGHLIGHT_COLOR.get(label, HIGHLIGHT_COLOR["copied"])
            alpha = float(HIGHLIGHT_ALPHA.get(label, 0.25))
            x0 = float(rect["x0"])
            x1 = float(rect["x1"])
            y0 = float(rect["y0"])
            y1 = float(rect["y1"])
            width = max(0.8, x1 - x0)
            height = max(0.8, y1 - y0)

            overlay_canvas.saveState()
            if hasattr(overlay_canvas, "setFillAlpha"):
                overlay_canvas.setFillAlpha(alpha)
            overlay_canvas.setFillColorRGB(*color)
            overlay_canvas.rect(x0, y0, width, height, fill=1, stroke=0)
            overlay_canvas.restoreState()

        overlay_canvas.showPage()
    overlay_canvas.save()

    overlay_reader = PdfReader(str(overlay_pdf))
    writer = PdfWriter()
    for page_index, page in enumerate(source_reader.pages):
        if page_index < len(overlay_reader.pages):
            page.merge_page(overlay_reader.pages[page_index])
        writer.add_page(page)
    with output_pdf.open("wb") as handle:
        writer.write(handle)

    try:
        overlay_pdf.unlink(missing_ok=True)
    except Exception:
        pass
    return True


def _draw_wrapped_lines(
    pdf_canvas: canvas.Canvas,
    lines: Iterable[str],
    x_start: float,
    y_start: float,
    max_width_chars: int = 95,
    line_height: float = 14,
) -> float:
    """Draw wrapped text lines and return current y position."""
    y = y_start
    for raw_line in lines:
        chunks = textwrap.wrap(str(raw_line), width=max_width_chars) or [""]
        for chunk in chunks:
            pdf_canvas.drawString(x_start, y, chunk)
            y -= line_height
            if y <= 0.8 * inch:
                pdf_canvas.showPage()
                y = LETTER[1] - 0.9 * inch
    return y


def _text_to_pdf(text: str, output_path: Path, title: str) -> Path:
    """Render plain text to a PDF file."""
    pdf_canvas = canvas.Canvas(str(output_path), pagesize=LETTER)
    width, height = LETTER
    margin = 0.8 * inch
    y = height - margin

    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(margin, y, title[:120])
    y -= 22
    pdf_canvas.setFont("Helvetica", 10)

    text_lines = (text or "").splitlines()
    _draw_wrapped_lines(pdf_canvas, text_lines, x_start=margin, y_start=y, max_width_chars=95, line_height=13)
    pdf_canvas.save()
    return output_path


def _try_docx2pdf(input_path: Path, output_path: Path) -> bool:
    """Try converting doc/docx to PDF via docx2pdf when installed."""
    try:
        from docx2pdf import convert  # pragma: no cover - optional dependency

        convert(str(input_path), str(output_path))
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _try_libreoffice_convert(input_path: Path, output_dir: Path) -> Optional[Path]:
    """Try converting Office documents to PDF via libreoffice headless."""
    commands = [
        ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(input_path)],
        ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(input_path)],
    ]
    target_pdf = output_dir / f"{input_path.stem}.pdf"
    for cmd in commands:
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=120)
        except Exception:
            continue
        if target_pdf.exists() and target_pdf.stat().st_size > 0:
            return target_pdf
    return None


def _source_document_to_pdf(
    uploaded_file: FileStorage | None,
    temp_dir: Path,
    fallback_text: str,
) -> Optional[Path]:
    """Convert uploaded source document to PDF while preserving original format when possible."""
    if uploaded_file is None:
        return _text_to_pdf(fallback_text, temp_dir / "target_text.pdf", title="Submitted Text")

    filename = uploaded_file.filename or "document.txt"
    extension = Path(filename).suffix.lower()
    content = uploaded_file.read()
    uploaded_file.seek(0)

    if not content:
        return _text_to_pdf(fallback_text, temp_dir / "target_text.pdf", title="Submitted Text")

    input_path = temp_dir / f"source{extension or '.txt'}"
    input_path.write_bytes(content)

    if extension == ".pdf":
        return input_path

    if extension in DOC_EXTENSIONS:
        output_pdf = temp_dir / "converted_source.pdf"
        if _try_docx2pdf(input_path, output_pdf):
            return output_pdf
        converted = _try_libreoffice_convert(input_path, temp_dir)
        if converted:
            if converted != output_pdf:
                shutil.copyfile(converted, output_pdf)
            return output_pdf

    if extension in TEXT_EXTENSIONS:
        decoded = content.decode("utf-8", errors="ignore")
        return _text_to_pdf(decoded, temp_dir / "source_text.pdf", title=filename)

    # Fallback extraction for unknown or failed conversion cases.
    extracted_text, _pages, _meta = extract_text(str(input_path), uploaded_file.content_type or "")
    if extracted_text.strip():
        return _text_to_pdf(extracted_text, temp_dir / "source_extracted.pdf", title=filename)
    return _text_to_pdf(fallback_text, temp_dir / "target_text.pdf", title="Submitted Text")


def _summary_lines(report: Dict[str, object]) -> List[str]:
    """Build summary lines to render on the final analysis page."""
    ai_explanation = report.get("ai_explanation", {}) or {}
    top_features = ai_explanation.get("top_contributing_features", []) or []
    sentence_rows = report.get("sentence_highlighting", []) or []

    copied = sum(1 for item in sentence_rows if item.get("label") == "copied")
    paraphrased = sum(1 for item in sentence_rows if item.get("label") == "paraphrased")
    ai_likely = sum(1 for item in sentence_rows if item.get("label") == "ai_likely")
    unique = sum(1 for item in sentence_rows if item.get("label") == "unique")

    lines = [
        f"Analysis Mode: {str(report.get('analysis_mode', 'full')).upper()}",
        f"Total Similarity: {float(report.get('total_similarity', 0.0)):.2f}%",
        f"Exact Plagiarism: {float(report.get('exact_plagiarism', 0.0)):.2f}%",
        f"Paraphrased Content: {float(report.get('paraphrased_content', 0.0)):.2f}%",
        f"AI Likelihood: {float(report.get('ai_likelihood', 0.0)):.2f}%",
        f"Confidence: {float(report.get('confidence', 0.0)) * 100.0:.2f}%",
        f"Reference Sources Used: {int(report.get('reference_sources_used', 0))}",
        "",
        "Sentence Classification Counts:",
        f"- Copied: {copied}",
        f"- Paraphrased: {paraphrased}",
        f"- AI-Likely: {ai_likely}",
        f"- Unique: {unique}",
        "",
        "Top AI Contributing Features:",
    ]

    for row in top_features[:12]:
        feature = str(row.get("feature", "unknown"))
        impact = float(row.get("impact", 0.0))
        lines.append(f"- {feature}: {impact:.6f}")

    citation_data = report.get("citation_adjustment", {}) or {}
    lines.extend(
        [
            "",
            "Citation Adjustment:",
            f"- Citations Detected: {int(citation_data.get('citation_count', 0))}",
            f"- Citation Density: {float(citation_data.get('citation_density', 0.0)):.4f}",
        ]
    )
    return lines


def _build_summary_pdf(output_path: Path, report: Dict[str, object], original_file_name: str) -> Path:
    """Create summary PDF page from analysis report fields."""
    pdf_canvas = canvas.Canvas(str(output_path), pagesize=LETTER)
    width, height = LETTER
    margin = 0.8 * inch
    y = height - margin

    pdf_canvas.setFont("Helvetica-Bold", 15)
    pdf_canvas.drawString(margin, y, "Analysis Summary")
    y -= 20
    pdf_canvas.setFont("Helvetica", 10)
    pdf_canvas.drawString(margin, y, f"Original File: {original_file_name}")
    y -= 16
    pdf_canvas.drawString(margin, y, f"Report Version: {report.get('report_version', '1.0.0')}")
    y -= 18

    lines = _summary_lines(report)
    _draw_wrapped_lines(pdf_canvas, lines, x_start=margin, y_start=y, max_width_chars=95, line_height=13)
    pdf_canvas.save()
    return output_path


def export_analysis_pdf(
    uploaded_file: FileStorage | None,
    report: Dict[str, object],
    reports_dir: Path,
    file_tag: str,
    fallback_text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Export merged source document + final analysis summary page as PDF."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = _safe_name(file_tag)
    output_path = reports_dir / f"{safe_tag}_analysis.pdf"

    try:
        with TemporaryDirectory(prefix="ps_pdf_export_") as temp_name:
            temp_dir = Path(temp_name)
            source_pdf = _source_document_to_pdf(uploaded_file, temp_dir, fallback_text=fallback_text)
            highlighted_pdf = temp_dir / "highlighted_source.pdf"
            merged_source_pdf = source_pdf

            if source_pdf and source_pdf.exists():
                try:
                    if _apply_highlight_overlay(source_pdf, report, highlighted_pdf):
                        merged_source_pdf = highlighted_pdf
                except Exception:
                    merged_source_pdf = source_pdf

            summary_pdf = _build_summary_pdf(
                temp_dir / "analysis_summary.pdf",
                report=report,
                original_file_name=(uploaded_file.filename if uploaded_file and uploaded_file.filename else "pasted_text"),
            )

            writer = PdfWriter()
            if merged_source_pdf and merged_source_pdf.exists():
                source_reader = PdfReader(str(merged_source_pdf))
                for page in source_reader.pages:
                    writer.add_page(page)

            summary_reader = PdfReader(str(summary_pdf))
            for page in summary_reader.pages:
                writer.add_page(page)

            with output_path.open("wb") as handle:
                writer.write(handle)

        return str(output_path), None
    except Exception as exc:  # pragma: no cover - runtime defensive path
        return None, str(exc)
