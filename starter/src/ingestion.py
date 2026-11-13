"""Read and normalize PDF pages while preserving source and page."""

import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

logger = logging.getLogger(__name__)


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    *,
    start_page: int = 1,
    end_page: int | None = None,
) -> list[DocumentPage]:
    """Read physical PDF pages ``start_page`` through ``end_page`` (inclusive).

    Use ``end_page=76`` for the selected Python book coverage (printed page 66).
    None reads the entire PDF. Empty pages are retained and logged;
    this function does not count eligible content pages or perform OCR.
    """
    if end_page is not None and (type(end_page) is not int or end_page < 1):
        raise ValueError("end_page must be a positive integer or None")

    with Path(path).open("rb") as pdf_file:
        reader = PdfReader(pdf_file)
        page_count = len(reader.pages)
        selected_end = page_count if end_page is None else end_page
        if selected_end > page_count:
            raise ValueError(
                f"end_page={selected_end} exceeds the PDF's {page_count} pages"
            )

        pages: list[DocumentPage] = []
        for page_number in range(start_page, selected_end + 1):
            try:
                raw_text = reader.pages[page_number - 1].extract_text(
                    extraction_mode="layout"
                )
                text = normalize_text(raw_text or "")
            except Exception:
                logger.exception(
                    "PDF extraction failed: source=%s page=%d", source, page_number
                )
                raise

            if not text.strip():
                logger.warning(
                    "PDF page has no extracted text: source=%s page=%d",
                    source,
                    page_number,
                )

            pages.append(
                DocumentPage(
                    document_id=document_id,
                    source=source,
                    page=page_number,
                    text=text,
                )
            )

    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace, retaining indentation and paragraphs."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        indentation = line[: len(line) - len(line.lstrip(" \t"))]
        content = line[len(indentation) :]
        lines.append(indentation + re.sub(r"[^\S\n]+", " ", content))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
