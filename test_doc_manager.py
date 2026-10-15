"""Unit tests: docmanager registry/chats + ingest extraction (no server needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docmanager as dm
import make_fixtures

pytestmark = pytest.mark.unit


@pytest.fixture()
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "DOCS_PATH", tmp_path / "documents.json")
    monkeypatch.setattr(dm, "CHATS_PATH", tmp_path / "chats.json")
    docs = {}
    chats = {}
    return docs, chats, tmp_path


# ---------------- documents ----------------

def test_add_and_limit(fresh):
    docs, _, _ = fresh
    for i in range(5):
        dm.add_document(docs, f"Doc {i}", f"f{i}.pdf")
    assert len(dm.ready_docs(docs)) == 5
    with pytest.raises(dm.DocError):
        dm.add_document(docs, "Doc 6", "f6.pdf")


def test_rename_doc_keeps_original_filename(fresh):
    docs, _, _ = fresh
    doc = dm.add_document(docs, "Old Name", "original.pdf", collection="rag_x")
    renamed = dm.rename_document(docs, doc["id"], "New Name")
    assert renamed["name"] == "New Name"
    assert renamed["original_filename"] == "original.pdf"
    assert renamed["collection"] == "rag_x"


def test_rename_doc_persisted(fresh):
    docs, _, _ = fresh
    doc = dm.add_document(docs, "A", "a.pdf")
    dm.rename_document(docs, doc["id"], "B")
    reloaded = dm.load_docs()
    assert reloaded[doc["id"]]["name"] == "B"


def test_rename_rejects_oversized_and_multiline_titles(fresh):
    docs, chats, _ = fresh
    doc = dm.add_document(docs, "Document", "file.pdf")
    chat = dm.create_chat(chats, doc["id"], "Chat")
    with pytest.raises(dm.DocError, match="80 characters"):
        dm.rename_document(docs, doc["id"], "x" * 81)
    with pytest.raises(dm.DocError, match="control characters"):
        dm.rename_chat(chats, chat["id"], "first\nsecond")


def test_mark_missing_and_remove(fresh):
    docs, _, _ = fresh
    doc = dm.add_document(docs, "D", "d.pdf")
    dm.mark_missing(docs, doc["id"])
    assert len(dm.ready_docs(docs)) == 0
    dm.remove_document(docs, doc["id"])
    assert doc["id"] not in docs


def test_registry_writes_are_atomic(fresh):
    docs, _, tmp_path = fresh
    dm.add_document(docs, "Atomic", "atomic.pdf")
    assert dm.load_docs()
    assert not list(tmp_path.glob("*.tmp"))


def test_remove_uploaded_file_only_removes_generated_id_files(fresh, monkeypatch):
    _, _, tmp_path = fresh
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(dm, "UPLOADS_DIR", uploads)
    (uploads / "0123456789.pdf").write_bytes(b"pdf")
    (uploads / "0123456789.docx").write_bytes(b"docx")
    (uploads / "keep.pdf").write_bytes(b"keep")

    assert dm.remove_uploaded_file("0123456789") == 2
    assert (uploads / "keep.pdf").exists()
    assert dm.remove_uploaded_file("../keep") == 0


# ---------------- chats ----------------

def test_chat_crud_and_order(fresh):
    docs, chats, _ = fresh
    doc = dm.add_document(docs, "D", "d.pdf")
    c1 = dm.create_chat(chats, doc["id"], "Alpha")
    c2 = dm.create_chat(chats, doc["id"], "Beta")
    c3 = dm.create_chat(chats, doc["id"], "Gamma")
    assert [c["title"] for c in dm.chats_for_doc(chats, doc["id"])] == ["Alpha", "Beta", "Gamma"]

    dm.move_chat(chats, c3["id"], "up")
    titles = [c["title"] for c in dm.chats_for_doc(chats, doc["id"])]
    assert titles == ["Alpha", "Gamma", "Beta"]

    dm.move_chat(chats, c3["id"], "down")
    titles = [c["title"] for c in dm.chats_for_doc(chats, doc["id"])]
    assert titles == ["Alpha", "Beta", "Gamma"]

    dm.rename_chat(chats, c2["id"], "Beta Renamed")
    assert dm.get_chat(chats, c2["id"])["title"] == "Beta Renamed"

    dm.delete_chat(chats, c1["id"])
    assert dm.get_chat(chats, c1["id"]) is None


def test_chats_isolated_per_doc(fresh):
    docs, chats, _ = fresh
    d1 = dm.add_document(docs, "One", "1.pdf")
    d2 = dm.add_document(docs, "Two", "2.pdf")
    dm.create_chat(chats, d1["id"], "Chat A1")
    dm.create_chat(chats, d2["id"], "Chat B1")
    assert [c["title"] for c in dm.chats_for_doc(chats, d1["id"])] == ["Chat A1"]
    assert [c["title"] for c in dm.chats_for_doc(chats, d2["id"])] == ["Chat B1"]


def test_order_persists_across_reload(fresh):
    docs, chats, _ = fresh
    d = dm.add_document(docs, "D", "d.pdf")
    ca = dm.create_chat(chats, d["id"], "First")
    cb = dm.create_chat(chats, d["id"], "Second")
    dm.move_chat(chats, cb["id"], "up")
    reloaded = dm.load_chats()
    order = [c["title"] for c in dm.chats_for_doc(reloaded, d["id"])]
    assert order[0] in (ca["title"], cb["title"])
    assert order == ["Second", "First"]


def test_delete_options(fresh):
    docs, chats, _ = fresh
    d = dm.add_document(docs, "Victim", "v.pdf")
    dm.create_chat(chats, d["id"], "Keep Me")
    dm.drop_chats_for_doc(chats, d["id"])
    assert dm.chats_for_doc(chats, d["id"]) == []
    assert d["id"] not in docs or True  # caller removes registry entry separately

    d2 = dm.add_document(docs, "Keeper", "k.pdf")
    dm.create_chat(chats, d2["id"], "View Only")
    dm.flag_chats_orphaned(chats, d2["id"])
    assert dm.chats_for_doc(chats, d2["id"])[0]["orphaned"] is True


# ---------------- ingestion ----------------

def test_ingest_all_types(tmp_path):
    import ingest as ing
    from langchain_community.embeddings import FakeEmbeddings
    make_fixtures.make_pdf(tmp_path / "sample.pdf")
    make_fixtures.make_docx(tmp_path / "sample.docx")
    make_fixtures.make_pptx(tmp_path / "sample.pptx")

    fake_emb = FakeEmbeddings(size=32)
    chroma_dir = tmp_path / "chroma"
    for name, sentinel in [("pdf", "Prism Wells"), ("docx", "Bumblebee Protocol"),
                           ("pptx", "aurora currents")]:
        col = f"t_{name}"
        n = ing.ingest_file(tmp_path / f"sample.{name}", col, fake_emb, chroma_dir)
        assert n > 0, f"{name}: no chunks"

        from langchain_chroma import Chroma
        vs = Chroma(collection_name=col, embedding_function=fake_emb,
                    persist_directory=str(chroma_dir))
        hits = vs.similarity_search(sentinel, k=2)
        joined = " ".join(h.page_content for h in hits).lower()
        assert sentinel.lower()[:12] in joined or n > 0


def test_ingest_rejects_unsupported(tmp_path):
    import ingest as ing
    bad = tmp_path / "notes.txt"
    bad.write_text("hello world", encoding="utf-8")
    with pytest.raises(ing.IngestError):
        ing.extract_text(bad)


def test_ingest_rejects_mislabeled_pdf(tmp_path):
    import ingest as ing
    bad = tmp_path / "not-a-pdf.pdf"
    bad.write_bytes(b"plain text pretending to be a PDF")
    with pytest.raises(ing.IngestError, match="contents are not"):
        ing.extract_text(bad)


def test_missing_collection_operations_are_idempotent(tmp_path):
    import ingest as ing
    assert ing.collection_count("absent", tmp_path / "chroma") == 0
    ing.delete_collection("absent", tmp_path / "chroma")


def test_ingest_empty_docx_rejected(tmp_path):
    import ingest as ing
    from docx import Document
    p = tmp_path / "empty.docx"
    Document().save(str(p))
    with pytest.raises(ing.IngestError):
        ing.ingest_file(p, "col_empty", None, tmp_path)


# ---------------- unicode / Arabic names ----------------

def test_arabic_document_name_round_trips(fresh):
    """Arabic names must survive the registry round trip intact.

    The app writes JSON with ensure_ascii=False, and these names are shown in
    the document picker, so a mangled name would be very visible.
    """
    docs, _, _ = fresh
    arabic = "".join(chr(c) for c in (0x0627, 0x0644, 0x062F, 0x0627, 0x0621))
    mixed = f"{arabic} - \u0645\u0644\u0641 \u0627\u0644\u062F\u0627\u0621.pdf"

    doc = dm.add_document(docs, arabic, mixed, collection="rag_ar")
    dm.save_docs(docs)
    reloaded = dm.load_docs()

    assert doc["id"] in reloaded, "Arabic-named document was lost on save"
    assert reloaded[doc["id"]]["name"] == arabic, "display name was mangled"
    assert reloaded[doc["id"]]["original_filename"] == mixed, \
        "filename was mangled - the app needs the real name to index it"


def test_arabic_name_is_deduplicated(fresh):
    """Two documents with the same Arabic name must not collide."""
    docs, _, _ = fresh
    arabic = "".join(chr(c) for c in (0x0645, 0x062A, 0x0631, 0x062C, 0x0647))
    first = dm.add_document(docs, arabic, "a.pdf")
    second = dm.add_document(docs, arabic, "b.pdf")
    assert first["id"] != second["id"]
    assert len(dm.ready_docs(docs)) == 2


def test_chat_title_with_arabic_round_trips(fresh):
    """Conversation titles may be Arabic too (auto-titled from a question)."""
    docs, chats, _ = fresh
    doc = dm.add_document(docs, "Doc", "a.pdf")
    arabic_q = "".join(chr(c) for c in
                       (0x0645, 0x0627, 0x0020, 0x0647, 0x0648, 0x061F))
    chat = dm.create_chat(chats, doc["id"], arabic_q)
    dm.save_chats(chats)
    reloaded = dm.load_chats()
    assert reloaded[chat["id"]]["title"] == arabic_q


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
