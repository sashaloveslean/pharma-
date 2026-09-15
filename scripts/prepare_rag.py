#!/usr/bin/env python3
"""Extract dissolution-related chunks and optionally embed them into Chroma."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DISSOLUTION_TERMS = [
    "раствор",
    "растворение",
    "растворимость",
    "растворим",
    "растворитель",
    "подвижная фаза",
    "проба",
    "стандартный раствор",
    "испытуемый раствор",
    "solubility",
    "dissolution",
    "dissolve",
    "dissolved",
    "soluble",
    "insoluble",
    "solvent",
    "diluent",
    "mobile phase",
    "условия хроматографирования",
    "хроматографические условия",
    "скорость потока",
    "объем вводимой пробы",
    "объём вводимой пробы",
    "температура колонки",
    "column temperature",
    "flow rate",
    "sample solution",
    "standard solution",
    "test solution",
    "stock solution",
]

HEADING_RE = re.compile(
    r"^\s*((\d+(\.\d+){0,4})\s+)?"
    r"([A-ZА-Я][A-ZА-ЯA-Za-zА-Яа-я0-9 ,;:/()\\-]{3,120})\s*$"
)


@dataclass
class Paragraph:
    text: str
    source: str
    index: int
    page: int | None = None
    section: str | None = None


class PageTextTimeout(RuntimeError):
    pass


@contextmanager
def page_timeout(seconds: int):
    def _handle_timeout(signum, frame):
        raise PageTextTimeout(f"page text extraction exceeded {seconds}s")

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handle_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf(path: Path) -> list[Paragraph]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to read PDFs: pip install -r requirements.txt") from exc

    paragraphs: list[Paragraph] = []
    reader = PdfReader(str(path))
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            with page_timeout(10):
                text = normalize_text(page.extract_text() or "")
        except (PageTextTimeout, Exception) as exc:
            print(f"Warning: skipped {path} page {page_index}: {exc}")
            continue
        for block in split_paragraphs(text):
            paragraphs.append(
                Paragraph(
                    text=block,
                    source=str(path),
                    index=len(paragraphs),
                    page=page_index,
                )
            )

    return paragraphs


def read_pdf_ocr(
    path: Path,
    dpi: int = 220,
    engine: str = "auto",
    page_numbers: set[int] | None = None,
) -> list[Paragraph]:
    try:
        import fitz
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Install OCR dependencies to read scanned PDFs: "
            "pip install pymupdf pillow"
        ) from exc

    paragraphs: list[Paragraph] = []
    document = fitz.open(path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    ocr_engine = resolve_ocr_engine(engine)

    for page_index, page in enumerate(document, start=1):
        if page_numbers is not None and page_index not in page_numbers:
            continue
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        try:
            text = normalize_text(ocr_image(image, engine=ocr_engine))
        except (PageTextTimeout, Exception) as exc:
            print(f"Warning: skipped OCR {path} page {page_index}: {exc}", flush=True)
            continue
        for block in split_paragraphs(text):
            paragraphs.append(
                Paragraph(
                    text=block,
                    source=str(path),
                    index=len(paragraphs),
                    page=page_index,
                )
            )

    return paragraphs


def resolve_ocr_engine(engine: str) -> str:
    if engine != "auto":
        return engine
    if shutil.which("tesseract"):
        return "tesseract"
    return "ocrmac"


def ocr_image(image: "Image.Image", engine: str) -> str:
    if engine == "tesseract":
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            image.save(image_file.name)
            try:
                result = subprocess.run(
                    ["tesseract", image_file.name, "stdout", "-l", "rus+eng", "--psm", "6"],
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired as exc:
                raise PageTextTimeout("OCR exceeded 30s") from exc
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Tesseract OCR failed")
        return result.stdout

    if engine == "ocrmac":
        try:
            from ocrmac import ocrmac
        except ImportError as exc:
            raise RuntimeError("Install ocrmac or use Tesseract OCR") from exc
        lines = ocrmac.OCR(
            image,
            recognition_level="accurate",
            language_preference=["ru-RU", "en-US"],
            confidence_threshold=0.2,
        ).recognize()
        return "\n".join(line[0] for line in lines)

    raise ValueError(f"Unsupported OCR engine: {engine}")


def read_docx(path: Path) -> list[Paragraph]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Install python-docx to read DOCX: pip install -r requirements.txt") from exc

    document = Document(str(path))
    paragraphs: list[Paragraph] = []
    for paragraph in document.paragraphs:
        text = normalize_text(paragraph.text)
        if text:
            paragraphs.append(Paragraph(text=text, source=str(path), index=len(paragraphs)))
    return paragraphs


def read_text(path: Path) -> list[Paragraph]:
    text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    return [
        Paragraph(text=block, source=str(path), index=index)
        for index, block in enumerate(split_paragraphs(text))
    ]


def split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n|(?<=[.;:])\s*\n(?=[A-ZА-Я0-9])", text)
    return [normalize_text(block) for block in blocks if normalize_text(block)]


def read_document(
    path: Path,
    ocr: bool = True,
    ocr_dpi: int = 220,
    ocr_engine: str = "auto",
) -> list[Paragraph]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        paragraphs = read_pdf(path)
        if ocr:
            try:
                import fitz
            except ImportError as exc:
                raise RuntimeError("Install pymupdf for per-page OCR fallback") from exc
            with fitz.open(path) as document:
                all_pages = set(range(1, document.page_count + 1))
            text_by_page: dict[int, int] = {}
            for paragraph in paragraphs:
                if paragraph.page is not None:
                    text_by_page[paragraph.page] = text_by_page.get(paragraph.page, 0) + len(paragraph.text)
            sparse_pages = {page for page in all_pages if text_by_page.get(page, 0) < 80}
            if sparse_pages:
                ocr_paragraphs = read_pdf_ocr(
                    path,
                    dpi=ocr_dpi,
                    engine=ocr_engine,
                    page_numbers=sparse_pages,
                )
                # Replace sparse text-layer output with OCR for those pages.
                paragraphs = [p for p in paragraphs if p.page not in sparse_pages] + ocr_paragraphs
                paragraphs.sort(key=lambda p: (p.page or 0, p.index))
                for index, paragraph in enumerate(paragraphs):
                    paragraph.index = index
        return paragraphs
    if suffix == ".docx":
        return read_docx(path)
    if suffix in {".txt", ".md"}:
        return read_text(path)
    raise ValueError(f"Unsupported file type: {path}")


def is_heading(text: str) -> bool:
    if len(text) > 140:
        return False
    return bool(HEADING_RE.match(text)) or text.endswith(":")


def term_hits(text: str) -> list[str]:
    lower = text.lower()
    return [term for term in DISSOLUTION_TERMS if term in lower]


def annotate_sections(paragraphs: list[Paragraph]) -> list[Paragraph]:
    current_section = None
    for paragraph in paragraphs:
        if is_heading(paragraph.text):
            current_section = paragraph.text.rstrip(":")
        paragraph.section = current_section
    return paragraphs


def select_relevant(paragraphs: list[Paragraph], context: int) -> list[Paragraph]:
    selected_indexes: set[int] = set()
    dissolution_section = False

    for index, paragraph in enumerate(paragraphs):
        hits = term_hits(paragraph.text)
        section_hits = term_hits(paragraph.section or "")

        if is_heading(paragraph.text):
            dissolution_section = bool(hits)

        if hits or section_hits or dissolution_section:
            start = max(0, index - context)
            end = min(len(paragraphs), index + context + 1)
            selected_indexes.update(range(start, end))

    return [paragraphs[index] for index in sorted(selected_indexes)]


def merge_into_chunks(paragraphs: list[Paragraph], max_chars: int, overlap_chars: int) -> list[dict]:
    chunks: list[dict] = []
    current: list[Paragraph] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return

        text = "\n\n".join(paragraph.text for paragraph in current)
        source = current[0].source
        chunk_id = hashlib.sha1(f"{source}:{current[0].index}:{text}".encode()).hexdigest()
        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "source": source,
                    "start_paragraph": current[0].index,
                    "end_paragraph": current[-1].index,
                    "start_page": current[0].page,
                    "end_page": current[-1].page,
                    "section": current[0].section,
                    "terms": sorted(set(term_hits(text))),
                },
            }
        )

        if overlap_chars <= 0:
            current = []
        else:
            overlap: list[Paragraph] = []
            size = 0
            for paragraph in reversed(current):
                size += len(paragraph.text)
                overlap.insert(0, paragraph)
                if size >= overlap_chars:
                    break
            current = overlap
        current_chars = sum(len(paragraph.text) for paragraph in current)

    for paragraph in paragraphs:
        paragraph_len = len(paragraph.text)
        if current and current_chars + paragraph_len > max_chars:
            flush()
        current.append(paragraph)
        current_chars += paragraph_len

    flush()
    return chunks


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    for suffix in ("*.pdf", "*.docx", "*.txt", "*.md"):
        yield from sorted(input_path.rglob(suffix))


def save_jsonl(
    chunks: list[dict],
    output_path: Path,
    replace_sources: set[str] | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = chunks
    if replace_sources is not None and output_path.exists():
        retained = []
        with output_path.open(encoding="utf-8") as existing:
            for line in existing:
                row = json.loads(line)
                if row.get("metadata", {}).get("source") not in replace_sources:
                    retained.append(row)
        combined = retained + chunks
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in combined:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return len(combined)


def embed_chroma(chunks: list[dict], persist_dir: Path, collection_name: str, model: str) -> None:
    try:
        import chromadb
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install chromadb and openai: pip install -r requirements.txt") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()
    chroma = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma.get_or_create_collection(name=collection_name)

    for start in range(0, len(chunks), 64):
        batch = chunks[start : start + 64]
        response = client.embeddings.create(model=model, input=[chunk["text"] for chunk in batch])
        embeddings = [item.embedding for item in response.data]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
            embeddings=embeddings,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="File or folder with PDF/DOCX/TXT/MD docs")
    parser.add_argument("--output", default="data/processed/dissolution_chunks.jsonl")
    parser.add_argument("--context", type=int, default=1, help="Neighbor paragraphs around each match")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--ocr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ocr-dpi", type=int, default=220)
    parser.add_argument("--ocr-engine", choices=["auto", "tesseract", "ocrmac"], default="auto")
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Replace chunks only for input files while retaining other sources in output",
    )
    parser.add_argument("--embed", action="store_true", help="Write embeddings into local Chroma")
    parser.add_argument("--persist-dir", default="vector_store")
    parser.add_argument("--collection", default="molecule_dissolution")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    args = parser.parse_args()

    input_path = Path(args.input)
    files = list(iter_input_files(input_path))
    if not files:
        raise SystemExit(f"No supported documents found in {input_path}")

    all_chunks: list[dict] = []
    for path in files:
        print(f"Reading {path}", flush=True)
        paragraphs = annotate_sections(
            read_document(
                path,
                ocr=args.ocr,
                ocr_dpi=args.ocr_dpi,
                ocr_engine=args.ocr_engine,
            )
        )
        relevant = select_relevant(paragraphs, context=args.context)
        all_chunks.extend(merge_into_chunks(relevant, args.max_chars, args.overlap_chars))

    total = save_jsonl(
        all_chunks,
        Path(args.output),
        replace_sources={str(path) for path in files} if args.update_existing else None,
    )
    print(f"Saved {total} chunks to {args.output} ({len(all_chunks)} from current input)")

    if args.embed:
        embed_chroma(
            chunks=all_chunks,
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
            model=args.embedding_model,
        )
        print(f"Embedded {len(all_chunks)} chunks into Chroma collection {args.collection}")


if __name__ == "__main__":
    main()
