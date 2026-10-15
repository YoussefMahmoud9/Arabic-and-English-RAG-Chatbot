"""Registry maintenance: prune unusable documents and orphaned artefacts.

Documents with failed or invalid status are stale. Records marked "missing"
are retained as history unless --all-missing is supplied.

Usage:
    python cleanup_registry.py --dry-run     # report only
    python cleanup_registry.py               # prune
    python cleanup_registry.py --all-missing # also drop "missing" history
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import docmanager as dm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DOCS_PATH = ROOT / ".tools" / "documents.json"
CHATS_PATH = ROOT / ".tools" / "chat_history.json"
UPLOADS_DIR = ROOT / ".tools" / "uploads"
CHROMA_DIR = ROOT / "chroma_db"


def load_docs() -> list[dict]:
    """Read the document registry as a list, or [] when it does not exist."""
    if not DOCS_PATH.exists():
        return []
    return json.loads(DOCS_PATH.read_text(encoding="utf-8")).get("docs", [])


def save_docs(docs: list[dict]) -> None:
    """Write the registry back, preserving the on-disk format."""
    dm._atomic_write(
        DOCS_PATH,
        json.dumps({"version": 1, "docs": docs}, ensure_ascii=False, indent=1),
    )


def drop_collection(name: str | None) -> None:
    """Delete a Chroma collection, ignoring the case where it is absent."""
    if not name:
        return
    try:
        import ingest as ing
        ing.delete_collection(name, CHROMA_DIR)
    except Exception as exc:
        raise RuntimeError(f"Could not delete collection {name!r}.") from exc


def main() -> int:
    """Report (and optionally prune) unusable registry entries.

    Returns a process exit code: 0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report only")
    ap.add_argument("--all-missing", action="store_true",
                    help="also prune entries marked missing")
    args = ap.parse_args()

    docs = load_docs()
    if not docs:
        print("No registry found.")
        return 0

    def unusable(d: dict) -> bool:
        """A document is unusable unless it is indexed and marked ready."""
        status = d.get("status")
        if status == "missing":
            return args.all_missing
        return status != "ready"

    keep = [d for d in docs if not unusable(d)]
    drop = [d for d in docs if unusable(d)]

    print(f"Registry: {len(docs)} document(s); {len(keep)} usable, {len(drop)} stale.")
    for d in drop:
        print(f"  - {d.get('name')!r} status={d.get('status')!r} "
              f"file={d.get('original_filename')!r}")

    # Report orphaned uploaded files (based on what will survive the prune).
    known_ids = {d.get("id") for d in keep}
    orphans = []
    if UPLOADS_DIR.exists():
        root = UPLOADS_DIR.resolve()
        orphans = [p for p in UPLOADS_DIR.iterdir()
                   if p.is_file() and not p.is_symlink()
                   and p.resolve().parent == root and p.stem not in known_ids]
        if orphans:
            print(f"Orphaned upload file(s): {len(orphans)}")
            for p in orphans[:10]:
                print(f"  - {p.name}")

    if not drop:
        if not args.dry_run:
            for p in orphans:
                p.unlink(missing_ok=True)
        print("Nothing to prune.")
        return 0

    if args.dry_run:
        print("\nDry run - nothing changed. Re-run without --dry-run to apply.")
        return 0

    try:
        for d in drop:
            drop_collection(d.get("collection"))
    except RuntimeError as exc:
        print(f"ERROR: {exc} Registry and uploads were not pruned.")
        return 1
    save_docs(keep)
    for p in orphans:
        p.unlink(missing_ok=True)

    # Chats pointing at a removed document are kept but marked orphaned.
    if CHATS_PATH.exists():
        cdata = json.loads(CHATS_PATH.read_text(encoding="utf-8"))
        chats = cdata.get("chats", [])
        live = {d.get("id") for d in keep}
        touched = 0
        for c in chats:
            if c.get("doc_id") not in live and not c.get("orphaned"):
                c["orphaned"] = True
                touched += 1
        if touched:
            dm._atomic_write(
                CHATS_PATH,
                json.dumps(cdata, ensure_ascii=False, indent=1),
            )
        print(f"Marked {touched} chat(s) as orphaned (history preserved).")

    print(f"\nPruned {len(drop)} document(s); {len(keep)} remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
