"""
RAG ingestion pipeline for a structured textbook PDF.
Pipeline: PDF -> chapter boundaries (bookmarks/outline) -> heading-aware
          line grouping -> section tree -> chunking (heading-based + size
          fallback) -> persist chunks to a local JSONL (for inspection).

Uses pdfplumber (text + font metadata, pure Python) and pypdf (TOC/outline,
pure Python) -- neither needs a compiled toolchain, so this installs cleanly
via `pip install -r requirements.txt` on any OS/Python version, no Visual
Studio / build-tools requirement.

This script is TEXT-ONLY by design (see mentor note in chat) -- image
extraction is a separate, later stage. Every chunk reserves an empty
`images` field so the schema doesn't need to change when that stage is added.

Usage:
    python ingest.py --pdf /path/to/book.pdf --out_dir ./output
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import tiktoken
from pypdf import PdfReader

ENC = tiktoken.get_encoding("cl100k_base")


def make_chunk_id(document_id: str, text: str) -> str:
    """Stable, content-derived chunk ID.

    - Prefixed with document_id so multiple source books/docs can share one
      vector DB without collisions, and so you can filter/delete by source.
    - The hash is computed from the final chunk TEXT (heading path + body,
      i.e. exactly what gets embedded) -- NOT from page number or position.
      This means: if you insert/delete pages elsewhere in the PDF and this
      chunk's text is byte-identical on re-ingest, it gets the SAME id, even
      though its page number changed. That's intentional -- see load_to_chroma.py,
      which uses this to skip re-embedding unchanged chunks and only refresh
      their (now-stale) page-number metadata.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{document_id}_{content_hash}"


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


# --------------------------------------------------------------------------
# 1. Chapter boundaries from the PDF's embedded outline/bookmarks
# --------------------------------------------------------------------------

@dataclass
class Chapter:
    title: str
    start_page: int  # 0-indexed, inclusive
    end_page: int    # 0-indexed, inclusive


def extract_chapters(pdf_path: str, total_pages: int) -> list[Chapter]:
    reader = PdfReader(pdf_path)
    outline = reader.outline
    if not outline:
        raise ValueError(
            "No embedded outline/bookmarks found. Fall back to manual "
            "chapter boundaries or a layout-based chapter-start detector."
        )

    skip_titles = {"cover", "back cover", "contents", "preface to the fourth edition",
                   "preface to the first edition"}

    chapters_raw = []
    for item in outline:
        if not hasattr(item, "title"):
            continue  # skip nested sub-outline lists for now (top-level only)
        title = item.title.strip()
        if title.lower() in skip_titles:
            continue
        page_num = reader.get_destination_page_number(item)  # 0-indexed
        chapters_raw.append((title, page_num))

    chapters = []
    for i, (title, start) in enumerate(chapters_raw):
        end = (chapters_raw[i + 1][1] - 1) if i + 1 < len(chapters_raw) else total_pages - 1
        chapters.append(Chapter(title=title, start_page=start, end_page=end))
    return chapters


# --------------------------------------------------------------------------
# 2. Line-level extraction with font metadata (needed for heading detection)
# --------------------------------------------------------------------------

@dataclass
class Line:
    text: str
    size: float
    bold: bool
    page: int  # 0-indexed


def extract_lines(page: "pdfplumber.page.Page", page_num: int) -> list[Line]:
    """Group pdfplumber words into visual lines using vertical position,
    then determine per-line size/bold from the words that make up that line."""
    words = page.extract_words(extra_attrs=["fontname", "size"], use_text_flow=False)
    if not words:
        return []

    # Group words whose 'top' (vertical position) is within a small
    # tolerance of each other into the same line.
    words_sorted = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    lines_raw: list[list[dict]] = []
    current_line: list[dict] = []
    current_top = None
    TOP_TOLERANCE = 2.0

    for w in words_sorted:
        if current_top is None or abs(w["top"] - current_top) <= TOP_TOLERANCE:
            current_line.append(w)
            current_top = w["top"] if current_top is None else current_top
        else:
            lines_raw.append(current_line)
            current_line = [w]
            current_top = w["top"]
    if current_line:
        lines_raw.append(current_line)

    lines = []
    for line_words in lines_raw:
        line_words.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line_words).strip()
        if not text:
            continue
        all_bold = all("Bold" in w["fontname"] for w in line_words)
        avg_size = sum(w["size"] for w in line_words) / len(line_words)
        lines.append(Line(text=text, size=round(avg_size, 1), bold=all_bold, page=page_num))
    return lines


# --------------------------------------------------------------------------
# 3. Header/footer stripping (running headers repeat almost verbatim/page)
# --------------------------------------------------------------------------

RUNNING_HEADER_PATTERNS = [
    re.compile(r"^\d+\s*I\s*Handbook of General Anatomy\s*$", re.IGNORECASE),
    re.compile(r"^[A-Za-z ,]+\s*I\s*\d+\s*$"),  # e.g. "Introduction I 19"
]


def is_running_header_or_footer(line: Line) -> bool:
    return any(p.match(line.text) for p in RUNNING_HEADER_PATTERNS)


# --------------------------------------------------------------------------
# 4. Heading detection heuristic
# --------------------------------------------------------------------------
# A line is treated as a SUBHEADING if:
#   - every word on the line is bold
#   - it's short (<= 8 words) -- real subheadings in this book are short
#     phrases like "In the Neck", "Terms Used for Describing Muscles"
#   - it's not just a bullet glyph or roman-numeral bullet artifact
#
# NOTE (mentor flag for you): this is a heuristic tuned to this book's
# layout. On noisier OCR you WILL get false positives/negatives -- treat
# this as a first pass, and spot check ~20 detected headings against the
# real book before trusting it at scale.

def is_heading_candidate(line: Line) -> bool:
    if not line.bold:
        return False
    words = line.text.split()
    if not (1 <= len(words) <= 8):
        return False
    if line.text.strip() in {"•", "-", "*"}:
        return False
    if re.match(r"^\(?[ivxIVX]+\)?[.)]?$", line.text.strip()):
        return False  # roman numeral bullets like "(i)"
    return True


# --------------------------------------------------------------------------
# 5. Build a section tree: Chapter -> Subheading -> body text
# --------------------------------------------------------------------------

@dataclass
class Section:
    chapter: str
    subheading: str  # "(intro)" if body text appears before any subheading
    page_start: int
    page_end: int
    text_lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines).strip()


def build_sections(pdf: "pdfplumber.PDF", chapter: Chapter) -> list[Section]:
    """NOTE on page numbers: page_num here is pdfplumber's 0-indexed internal
    page position. Section.page_start/page_end store the 1-INDEXED version
    (+1) -- matching what a "Go to page" box in any standard PDF viewer
    expects, so a citation like "page 129" actually jumps to the right page
    when a user opens this PDF and types 129. This is the standard citation
    convention for RAG-over-PDF (navigable in the viewer), not the book's own
    printed page label, which would require the reader to manually work out
    an offset before it's useful."""
    sections: list[Section] = []
    current = Section(chapter=chapter.title, subheading="(intro)",
                       page_start=chapter.start_page + 1, page_end=chapter.start_page + 1)

    for page_num in range(chapter.start_page, chapter.end_page + 1):
        page = pdf.pages[page_num]
        lines = extract_lines(page, page_num)
        for line in lines:
            if is_running_header_or_footer(line):
                continue
            if is_heading_candidate(line):
                if current.text:
                    sections.append(current)
                current = Section(chapter=chapter.title, subheading=line.text,
                                   page_start=page_num + 1, page_end=page_num + 1)
            else:
                current.text_lines.append(line.text)
                current.page_end = page_num + 1

    if current.text:
        sections.append(current)
    return sections


# --------------------------------------------------------------------------
# 6. Chunking: heading-based primary split, size-based fallback
# --------------------------------------------------------------------------

MIN_TOKENS = 100
MAX_TOKENS = 500
OVERLAP_TOKENS = 60  # ~10-15% of MAX_TOKENS, used only on forced splits


@dataclass
class Chunk:
    chunk_id: str
    chapter: str
    subheading: str
    page_start: int
    page_end: int
    text: str          # heading path + body, ready to embed
    token_count: int
    images: list = field(default_factory=list)  # populated in a later stage


def split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Recursively split on paragraph boundaries, then sentences, with overlap."""
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = re.split(r"(?<=[.!?])\s+", text)

    pieces, current, current_tokens = [], [], 0
    for para in paragraphs:
        t = count_tokens(para)
        if current_tokens + t > max_tokens and current:
            pieces.append(" ".join(current))
            overlap_text = " ".join(current)[-overlap_tokens * 4:]  # rough char proxy
            current = [overlap_text, para]
            current_tokens = count_tokens(" ".join(current))
        else:
            current.append(para)
            current_tokens += t
    if current:
        pieces.append(" ".join(current))
    return pieces


def sections_to_chunks(sections: list[Section], document_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: Section | None = None

    def flush(sec: Section):
        heading_path = f"{sec.chapter} > {sec.subheading}"
        body_pieces = split_long_text(sec.text, MAX_TOKENS, OVERLAP_TOKENS)
        for piece in body_pieces:
            full_text = f"{heading_path}\n\n{piece}"
            # chunk_id is derived from full_text alone (see make_chunk_id) --
            # NOT from chapter/index/page -- so re-running ingest.py after a
            # page-count change (inserted/removed pages elsewhere) gives the
            # SAME id to unchanged content, and only a NEW id to content that
            # actually changed. This is what load_to_chroma.py's upsert logic
            # relies on to skip re-embedding unchanged chunks.
            chunks.append(Chunk(
                chunk_id=make_chunk_id(document_id, full_text),
                chapter=sec.chapter,
                subheading=sec.subheading,
                page_start=sec.page_start,
                page_end=sec.page_end,
                text=full_text,
                token_count=count_tokens(full_text),
            ))

    for sec in sections:
        tok = count_tokens(sec.text)
        if tok < MIN_TOKENS:
            if buffer is None:
                buffer = sec
            else:
                buffer.text_lines.append(f"[{sec.subheading}] " + sec.text)
                buffer.page_end = sec.page_end
                buffer.subheading = f"{buffer.subheading} / {sec.subheading}"
            if buffer and count_tokens(buffer.text) >= MIN_TOKENS:
                flush(buffer)
                buffer = None
        else:
            if buffer is not None:
                flush(buffer)
                buffer = None
            flush(sec)

    if buffer is not None and buffer.text.strip():
        flush(buffer)

    return chunks


# --------------------------------------------------------------------------
# 7. Orchestration
# --------------------------------------------------------------------------

def run_pipeline(pdf_path: str, out_dir: str, document_id: str) -> list[Chunk]:
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    chapters = extract_chapters(pdf_path, total_pages)
    print(f"Found {len(chapters)} chapters.")

    all_chunks: list[Chunk] = []
    seen_ids: dict[str, int] = {}  # chunk_id -> count seen so far
    with pdfplumber.open(pdf_path) as pdf:
        for chapter in chapters:
            sections = build_sections(pdf, chapter)
            chunks = sections_to_chunks(sections, document_id)
            print(f"  {chapter.title!r}: {len(sections)} sections -> {len(chunks)} chunks "
                  f"(pages {chapter.start_page + 1}-{chapter.end_page + 1})")
            all_chunks.extend(chunks)

    # Collision guard: two DIFFERENT sections can (rarely) produce identical
    # chunk text (e.g. a short repeated phrase), which would hash to the
    # same chunk_id and silently overwrite each other in the vector DB.
    # If that happens, append a disambiguating suffix to all but the first.
    for c in all_chunks:
        seen_ids[c.chunk_id] = seen_ids.get(c.chunk_id, 0) + 1
        if seen_ids[c.chunk_id] > 1:
            original_id = c.chunk_id
            c.chunk_id = f"{c.chunk_id}_dup{seen_ids[c.chunk_id]}"
            print(f"  WARNING: duplicate chunk content detected ({original_id}); "
                  f"disambiguated as {c.chunk_id}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "chapter": c.chapter,
                "subheading": c.subheading,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
                "token_count": c.token_count,
                "images": c.images,
            }, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks to {jsonl_path}")
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out_dir", default="./output")
    parser.add_argument("--document_id", default="anatomy_bd_chaurasia_4thed",
                         help="Stable identifier for this source document. Prefixed onto "
                              "every chunk_id, so multiple books can share one vector DB "
                              "without ID collisions, and so you can filter/delete by source.")
    args = parser.parse_args()
    run_pipeline(args.pdf, args.out_dir, args.document_id)