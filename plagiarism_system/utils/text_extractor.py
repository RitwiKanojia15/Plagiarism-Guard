"""Text extraction helpers for uploaded files in plagiarism_system."""

from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path
from typing import Dict, Tuple
from xml.etree import ElementTree

from PyPDF2 import PdfReader


def estimate_pages(word_count: int) -> int:
    """Estimate page count for plain text sources."""
    if word_count <= 0:
        return 1
    return max(1, math.ceil(word_count / 400))


def text_stats(text: str) -> Dict[str, int]:
    """Compute basic text statistics."""
    words = re.findall(r"\b\w+\b", text or "")
    return {"words": len(words), "characters": len(text or "")}


def normalize_text(text: str) -> str:
    """Normalize whitespace for analysis consistency."""
    if not text:
        return ""
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"\u00a0", " ", cleaned)
    cleaned = re.sub(r"[\t\f\v]+", " ", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_pdf(path: str) -> Tuple[str, int]:
    reader = PdfReader(path)
    pages = len(reader.pages) or 1
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts), pages


def _extract_docx(path: str) -> str:
    # Minimal DOCX extraction using built-in zip/xml modules (no python-docx dependency).
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = []
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in archive.namelist():
            return ""
        with archive.open("word/document.xml") as handle:
            tree = ElementTree.parse(handle)
            for node in tree.findall(".//w:t", namespace):
                if node.text:
                    parts.append(node.text)
    return " ".join(parts)


def _extract_doc(path: str) -> str:
    # Legacy .doc fallback: decode bytes best-effort.
    raw = Path(path).read_bytes()
    return raw.decode("latin-1", errors="ignore")


def _extract_plain(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def extract_text(path: str, content_type: str = "") -> Tuple[str, int, Dict[str, object]]:
    """Extract text from supported file types.

    Returns: (text, pages, meta)
    """
    if not path:
        return "", 1, {"source": "none"}

    lower_path = path.lower()
    text = ""
    pages = 1
    source = "plain"

    if content_type == "application/pdf" or lower_path.endswith(".pdf"):
        source = "pdf"
        text, pages = _extract_pdf(path)

    elif (
        content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or lower_path.endswith(".docx")
    ):
        source = "docx"
        text = _extract_docx(path)
        pages = estimate_pages(text_stats(text)["words"])

    elif content_type == "application/msword" or (lower_path.endswith(".doc") and not lower_path.endswith(".docx")):
        source = "doc"
        text = _extract_doc(path)
        pages = estimate_pages(text_stats(text)["words"])

    else:
        text = _extract_plain(path)
        pages = estimate_pages(text_stats(text)["words"])

    normalized = normalize_text(text)
    return normalized, pages, {"source": source, "path": Path(path).name}
