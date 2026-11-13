"""Character and paragraph chunking with stable IDs and page offsets."""

import hashlib
import re

from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, start: int, end: int, method: str) -> Chunk:
    identity = (
        f"{page.document_id}:{page.page}:{method}:{start}:{end}:{page.text[start:end]}"
    )
    return Chunk(
        hashlib.sha256(identity.encode()).hexdigest(),
        page.document_id,
        page.source,
        page.page,
        page.text[start:end],
        {**page.metadata, "start_char": start, "end_char": end},
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    chunks = []
    for start in range(0, len(page.text), size - overlap):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            chunks.append(_chunk(page, start, end, "fixed"))
        if end == len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack paragraphs; split long paragraphs at whitespace where possible."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    chunks = []
    start, end = None, 0
    for paragraph in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", page.text):
        left, right = paragraph.span()
        if start is not None and right - start > max_size:
            chunks.append(_chunk(page, start, end, "paragraph"))
            start, end = None, 0
        while right - left > max_size:
            boundary = page.text.rfind(" ", left + 1, left + max_size + 1)
            cut = boundary if boundary > left else left + max_size
            chunks.append(_chunk(page, left, cut, "paragraph"))
            left = cut
            while left < right and page.text[left].isspace():
                left += 1
        if left < right:
            if start is None:
                start = left
            end = right
    if start is not None:
        chunks.append(_chunk(page, start, end, "paragraph"))
    return chunks
