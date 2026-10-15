"""Document registry + per-document chat store with persistence."""
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

MAX_DOCS = 5
MAX_TITLE_CHARS = 80
LEGACY_DOC_ID = "bigdata-legacy"
LEGACY_COLLECTION = "langchain"

PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_DIR = PROJECT_ROOT / ".tools"
DOCS_PATH = TOOLS_DIR / "documents.json"
CHATS_PATH = TOOLS_DIR / "chat_history.json"
UPLOADS_DIR = TOOLS_DIR / "uploads"


class DocError(Exception):
    pass


def _now():
    """Current wall-clock time, used for creation/ordering timestamps."""
    return time.time()


def _atomic_write(path, text):
    """Replace a JSON file atomically so interruption cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def load_docs():
    """Load the document registry, keyed by id.

    Returns an empty mapping when the file is missing or unreadable so a
    damaged file cannot stop the app from starting.
    """
    try:
        data = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
        return {d["id"]: d for d in data.get("docs", [])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {}


def save_docs(docs):
    """Write the document registry to disk, oldest document first.

    Storing it as an ordered list keeps the picker stable between runs.
    """
    ordered = sorted(docs.values(), key=lambda d: d.get("created", 0))
    _atomic_write(
        DOCS_PATH,
        json.dumps({"version": 1, "docs": ordered}, ensure_ascii=False, indent=1),
    )


def ready_docs(docs):
    """Documents that are indexed and can be asked questions against."""
    return [d for d in docs.values() if d.get("status") == "ready"]


def _validated_title(value, kind):
    """Normalize a user-visible document or chat title."""
    title = (value or "").strip()
    if not title:
        raise DocError(f"{kind} name cannot be empty.")
    if len(title) > MAX_TITLE_CHARS:
        raise DocError(f"{kind} name must be {MAX_TITLE_CHARS} characters or fewer.")
    if any(ord(char) < 32 for char in title):
        raise DocError(f"{kind} name cannot contain control characters.")
    return title


def add_document(docs, name, original_filename, collection=None, status="ready", embedding_model=None):
    """Register a new document and persist the registry.

    Args:
        docs: the live registry mapping (mutated in place).
        name: display name shown in the picker.
        original_filename: the uploaded file's name, kept for reference.
        collection: Chroma collection name; generated when omitted.
        status: "ready" once indexed, otherwise a failure marker.
        embedding_model: model used to embed it, so queries can match it.

    Returns:
        The new document record.

    Raises:
        DocError: when the document limit would be exceeded.
    """
    active = len(ready_docs(docs))
    if status == "ready" and active >= MAX_DOCS:
        raise DocError(
            f"Document limit reached ({active}/{MAX_DOCS}). "
            "Delete a document before adding another."
        )
    doc_id = uuid.uuid4().hex[:10]
    docs[doc_id] = {
        "id": doc_id,
        "name": name,
        "original_filename": original_filename,
        "collection": collection or f"rag_{doc_id}",
        "status": status,
        "created": _now(),
        "embedding_model": embedding_model,
    }
    save_docs(docs)
    return docs[doc_id]


def rename_document(docs, doc_id, new_name):
    """Rename a document, suffixing with a counter if the name is taken.

    Raises:
        DocError: for an empty name or an unknown document.
    """
    new_name = _validated_title(new_name, "Document")
    if doc_id not in docs:
        raise DocError("Document not found.")
    taken = {d["name"] for i, d in docs.items() if i != doc_id}
    final, n = new_name, 2
    while final in taken:
        final = f"{new_name} ({n})"
        n += 1
    docs[doc_id]["name"] = final
    save_docs(docs)
    return docs[doc_id]


def mark_missing(docs, doc_id):
    """Flag a document as no longer indexed (its file or index went away)."""
    if doc_id in docs:
        docs[doc_id]["status"] = "missing"
        save_docs(docs)


def remove_document(docs, doc_id):
    """Drop a document from the registry entirely."""
    docs.pop(doc_id, None)
    save_docs(docs)


def remove_uploaded_file(doc_id):
    """Delete the stored upload belonging to a generated document id."""
    if not re.fullmatch(r"[0-9a-f]{10}", str(doc_id or "")):
        return 0
    root = UPLOADS_DIR.resolve()
    removed = 0
    for path in UPLOADS_DIR.glob(f"{doc_id}.*"):
        try:
            if path.resolve().parent != root or path.is_symlink():
                continue
            path.unlink()
            removed += 1
        except OSError:
            raise
    return removed


# ---------------- chats ----------------

def load_chats():
    """Load stored conversations, migrating the pre-v2 title-keyed format.

    Version 1 files were a plain {title: body} mapping; those are converted
    to the current record format (with generated ids) and saved back once.
    """
    try:
        raw = json.loads(CHATS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    if isinstance(raw, dict) and raw.get("version") == 2:
        return {c["id"]: c for c in raw.get("chats", [])}
    migrated = {}
    if isinstance(raw, dict) and raw:
        for order, (title, body) in enumerate(raw.items()):
            cid = uuid.uuid4().hex[:10]
            migrated[cid] = {
                "id": cid,
                "doc_id": LEGACY_DOC_ID,
                "title": title,
                "order": order,
                "created": body.get("updated", _now()),
                "updated": body.get("updated", _now()),
                "messages": body.get("messages", []),
            }
        save_chats(migrated)
    return migrated


def save_chats(chats):
    """Persist conversations, most recently updated last."""
    ordered = sorted(chats.values(), key=lambda c: (c.get("updated", 0),))
    _atomic_write(
        CHATS_PATH,
        json.dumps({"version": 2, "chats": ordered}, ensure_ascii=False, indent=1),
    )


def chats_for_doc(chats, doc_id):
    """Conversations belonging to one document, in display order."""
    mine = [c for c in chats.values() if c.get("doc_id") == doc_id]
    mine.sort(key=lambda c: c.get("order", 0))
    return mine


def chat_titles(chats, doc_id):
    """Titles of a document's conversations, in display order."""
    return [c["title"] for c in chats_for_doc(chats, doc_id)]


def create_chat(chats, doc_id, title=None):
    """Add an empty conversation for a document and return it.

    Duplicate titles are disambiguated with a counter ("New Chat (2)") so the
    picker can use the title as a key.
    """
    existing = {c["title"] for c in chats_for_doc(chats, doc_id)}
    base = (title or "").strip() or "New Chat"
    final, n = base, 2
    while final in existing:
        final = f"{base} ({n})"
        n += 1
    orders = [c.get("order", 0) for c in chats_for_doc(chats, doc_id)]
    cid = uuid.uuid4().hex[:10]
    chats[cid] = {
        "id": cid,
        "doc_id": doc_id,
        "title": final,
        "order": (max(orders) + 1) if orders else 0,
        "created": _now(),
        "updated": _now(),
        "messages": [],
    }
    save_chats(chats)
    return chats[cid]


def get_chat(chats, chat_id):
    """Look up one conversation record, or None."""
    return chats.get(chat_id)


def update_chat_messages(chats, chat_id, messages):
    """Replace a conversation's messages and stamp it as just updated."""
    if chat_id in chats:
        chats[chat_id]["messages"] = messages
        chats[chat_id]["updated"] = _now()
        save_chats(chats)


def rename_chat(chats, chat_id, new_title):
    """Rename a conversation, suffixing a counter if the title is taken.

    Raises:
        DocError: when the conversation is unknown or the title is empty.
    """
    new_title = _validated_title(new_title, "Chat")
    chat = chats.get(chat_id)
    if not chat:
        raise DocError("Chat not found.")
    siblings = {c["title"] for c in chats_for_doc(chats, chat["doc_id"])
                if c["id"] != chat_id}
    final, n = new_title, 2
    while final in siblings:
        final = f"{new_title} ({n})"
        n += 1
    chat["title"] = final
    chat["updated"] = _now()
    save_chats(chats)
    return chat


def delete_chat(chats, chat_id):
    """Delete a conversation; unknown ids are ignored."""
    chats.pop(chat_id, None)
    save_chats(chats)


def move_chat(chats, chat_id, direction):
    """Swap a conversation with its neighbour ("up" or "down").

    Returns True when it moved, False when already at that end of the list.

    Raises:
        DocError: when the conversation is not part of that document's list.
    """
    chat = chats.get(chat_id)
    if not chat:
        raise DocError("Chat not found.")
    siblings = sorted(chats_for_doc(chats, chat["doc_id"]),
                      key=lambda c: c.get("order", 0))
    idx = next((i for i, c in enumerate(siblings) if c["id"] == chat_id), None)
    if idx is None:
        raise DocError("Chat not found in list.")
    swap = idx - 1 if direction == "up" else idx + 1
    if not 0 <= swap < len(siblings):
        return False
    siblings[idx], siblings[swap] = siblings[swap], siblings[idx]
    for pos, c in enumerate(siblings):
        c["order"] = pos
    save_chats(chats)
    return True


def drop_chats_for_doc(chats, doc_id):
    """Delete every conversation belonging to a document."""
    for cid in [i for i, c in chats.items() if c.get("doc_id") == doc_id]:
        chats.pop(cid, None)
    save_chats(chats)


def get_doc_embedding_model(docs, doc_id):
    """Get the embedding model for a specific document."""
    doc = docs.get(doc_id)
    if doc and doc.get("embedding_model"):
        return doc["embedding_model"]
    return None


def flag_chats_orphaned(chats, doc_id):
    """Mark a document's conversations as read-only history.

    Used when a document is deleted but its conversations are kept: they
    remain openable, but the UI blocks new questions against them.
    """
    changed = False
    for c in chats.values():
        if c.get("doc_id") == doc_id and not c.get("orphaned"):
            c["orphaned"] = True
            changed = True
    if changed:
        save_chats(chats)
