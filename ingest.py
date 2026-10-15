"""Document ingestion: extract text from PDF/DOCX/PPTX, chunk, embed into Chroma."""
from pathlib import Path
import os
import re
import zipfile
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class IngestError(Exception):
    pass


SUPPORTED = {".pdf", ".docx", ".pptx"}
MAX_UPLOAD_MB = max(1, int(os.environ.get("RAG_MAX_UPLOAD_MB", "40")))
MAX_FILE_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_EXTRACTED_CHARS = 8_000_000
MAX_PDF_PAGES = 3_000
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
OCR_ENABLED = os.environ.get("RAG_OCR", "0").strip().lower() in ("1", "true", "yes")
OCR_LANGUAGE = os.environ.get("RAG_OCR_LANG", "eng+ara")


def _validate_file(path: Path) -> None:
    """Check size and container structure before handing a file to parsers."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise IngestError("The uploaded file is unavailable.") from exc
    if not path.is_file():
        raise IngestError("The upload must be a regular file.")
    if size <= 0:
        raise IngestError("The uploaded file is empty.")
    if size > MAX_FILE_BYTES:
        raise IngestError(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")

    suffix = path.suffix.lower()
    try:
        with path.open("rb") as handle:
            header = handle.read(1024)
        if suffix == ".pdf":
            if b"%PDF-" not in header:
                raise IngestError("The file extension is PDF but its contents are not.")
            return
        if not zipfile.is_zipfile(path):
            raise IngestError("The file is not a valid DOCX or PPTX archive.")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise IngestError("The document contains too many archive entries.")
            expanded = sum(entry.file_size for entry in entries)
            if expanded > MAX_ARCHIVE_BYTES:
                raise IngestError("The document expands beyond the 200 MB safety limit.")
            if any(entry.flag_bits & 0x1 for entry in entries):
                raise IngestError("Password-protected Office documents are not supported.")
            names = {entry.filename for entry in entries}
            content_types = "[Content_Types].xml" in names
            expected = "word/document.xml" if suffix == ".docx" else "ppt/presentation.xml"
            if not content_types or expected not in names:
                raise IngestError("The file contents do not match its DOCX/PPTX extension.")
    except IngestError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise IngestError("The uploaded file is damaged or unreadable.") from exc


def _is_arabic_text(text: str) -> bool:
    """Detect if text contains substantial Arabic content."""
    if not text:
        return False
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    return arabic_chars >= 5 and arabic_chars > 0.12 * max(len(text), 1)


def _reshape_arabic(text: str) -> str:
    """Reshape Arabic text for proper display (ligatures + BiDi)."""
    if not _is_arabic_text(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def _extract_pdf(path: Path) -> list[str]:
    """Extract text from PDF using PyMuPDF (better Arabic support).

    Each page is tagged with a ``[Page N]`` marker so answers can cite where
    their evidence came from.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz  # Compatibility with older PyMuPDF releases.
        except ImportError:
            if OCR_ENABLED:
                raise IngestError(
                    "OCR was requested, but PyMuPDF is unavailable. Install "
                    "the project requirements or set RAG_OCR=0."
                )
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            if len(reader.pages) > MAX_PDF_PAGES:
                raise IngestError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
            texts = []
            char_count = 0
            for number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if _is_arabic_text(text):
                    text = _reshape_arabic(text)
                if text.strip():
                    texts.append(f"[Page {number}]\n{text}")
                    char_count += len(text)
                    if char_count > MAX_EXTRACTED_CHARS:
                        raise IngestError("Extracted text exceeds the 8 million character limit.")
            return texts

    texts = []
    char_count = 0
    with fitz.open(str(path)) as doc:
        if doc.page_count > MAX_PDF_PAGES:
            raise IngestError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            if OCR_ENABLED and len(text.strip()) < 20:
                try:
                    text_page = page.get_textpage_ocr(
                        language=OCR_LANGUAGE, dpi=200, full=True)
                    text = page.get_text("text", textpage=text_page, sort=True)
                except Exception as exc:
                    raise IngestError(
                        "OCR is enabled but Tesseract or its language data is "
                        f"unavailable ({type(exc).__name__}). Install Tesseract "
                        "with the configured language packs, or set RAG_OCR=0."
                    ) from exc
            if _is_arabic_text(text):
                text = _reshape_arabic(text)
            if text.strip():
                texts.append(f"[Page {number}]\n{text}")
                char_count += len(text)
                if char_count > MAX_EXTRACTED_CHARS:
                    raise IngestError("Extracted text exceeds the 8 million character limit.")
    return texts


def _extract_docx(path: Path) -> str:
    """Pull the text out of a Word document, one block per paragraph.

    Arabic paragraphs are reshaped so joined and right-to-left letters render
    correctly once the text reaches the model.
    """
    import docx2txt
    text = docx2txt.process(str(path))
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    if _is_arabic_text("\n".join(parts)):
        parts = [_reshape_arabic(p) for p in parts]
    return "\n\n".join(parts) if parts else ""


def _extract_pptx(path: Path) -> str:
    """Pull the text out of a PowerPoint deck, tagged by slide number.

    The ``[Slide N]`` tags become the source references shown under answers.
    """
    from pptx import Presentation
    prs = Presentation(str(path))
    chunks = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                t = shape.text_frame.text.strip()
                if t:
                    texts.append(t)
        if texts:
            if _is_arabic_text("\n".join(texts)):
                texts = [_reshape_arabic(t) for t in texts]
            chunks.append(f"[Slide {i}]\n" + "\n".join(texts))
    return "\n\n".join(chunks)


def extract_text(path: Path) -> str:
    """Return the readable text of a PDF, DOCX or PPTX file.

    Raises IngestError for unsupported types and for files that cannot be
    read (corrupt, encrypted, or containing no selectable text).
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise IngestError(
            f"Unsupported file type '{suffix}'. Supported: PDF, DOCX, PPTX."
        )
    try:
        _validate_file(path)
        if suffix == ".pdf":
            text = "\n\n".join(_extract_pdf(path))
        elif suffix == ".docx":
            text = _extract_docx(path)
        else:
            text = _extract_pptx(path)
        if len(text) > MAX_EXTRACTED_CHARS:
            raise IngestError("Extracted text exceeds the 8 million character limit.")
        return text
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(
            f"Could not read '{path.name}' ({type(exc).__name__}). "
            "The file may be corrupted or password-protected."
        ) from exc


def ingest_file(
    path: Path,
    collection_name: str,
    embedding,
    chroma_dir: Path,
    embedding_model_name: Optional[str] = None,
    text_content: Optional[str] = None,
) -> int:
    """Chunk a document and store the vectors in a Chroma collection.

    Chunking adapts to the language: Arabic uses smaller chunks with Arabic
    punctuation as separators, English uses larger ones.

    Args:
        path: the file to index.
        collection_name: Chroma collection to write into.
        embedding: LangChain-compatible embedding client.
        chroma_dir: on-disk location of the vector store.
        embedding_model_name: recorded in metadata so queries can reuse the
            same model later.

    Returns:
        The number of chunks stored.

    Raises:
        IngestError: for a missing/unsupported file, no extractable text, or
            an empty chunk set.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"File not found: {path.name}")
    if path.suffix.lower() not in SUPPORTED:
        raise IngestError(
            f"Unsupported file type '{path.suffix}'. Supported: PDF, DOCX, PPTX."
        )

    if text_content is None:
        raw = extract_text(path)
    else:
        _validate_file(path)
        raw = text_content
    if len(raw.strip()) < 20:
        raise IngestError(
            f"No readable text found in '{path.name}'. "
            "It may be empty or scanned. For scanned PDFs, install Tesseract "
            "with the needed language packs and set RAG_OCR=1 "
            "(RAG_OCR_LANG defaults to eng+ara), then retry."
        )

    doc = Document(
        page_content=raw,
        metadata={"source": path.name, "embedding_model": embedding_model_name}
    )

    # Determine if Arabic text needs special chunking
    is_arabic = _is_arabic_text(raw)
    if is_arabic:
        # Arabic text benefits from smaller chunks with less overlap
        # since Arabic words are often longer and context is denser
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", "۔", "،", "؛", ""],  # Arabic punctuation
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

    splits = splitter.split_documents([doc])
    if not splits:
        raise IngestError(f"Could not create any text chunks from '{path.name}'.")

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(chroma_dir),
    )
    vs.add_documents(splits)
    return len(splits)


def delete_collection(collection_name: str, chroma_dir: Path) -> None:
    """Remove a Chroma collection, ignoring only the missing-collection case."""
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(collection_name)
    except chromadb.errors.NotFoundError:
        return


def collection_count(collection_name: str, chroma_dir: Path) -> int:
    """Number of chunks in a collection, or 0 when it does not exist yet."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        col = client.get_collection(collection_name)
        return col.count()
    except chromadb.errors.NotFoundError:
        return 0
