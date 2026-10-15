"""Regression tests for registry pruning safety and flag behavior."""
import json
import sys

import pytest

import cleanup_registry as cleanup

pytestmark = pytest.mark.unit


@pytest.fixture
def registry_paths(tmp_path, monkeypatch):
    """Redirect cleanup paths into an isolated temporary directory."""
    docs = tmp_path / "documents.json"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(cleanup, "DOCS_PATH", docs)
    monkeypatch.setattr(cleanup, "CHATS_PATH", tmp_path / "chats.json")
    monkeypatch.setattr(cleanup, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(cleanup, "CHROMA_DIR", tmp_path / "chroma")
    return docs, uploads


def _write_registry(path, records):
    """Write minimal registry fixture data."""
    path.write_text(json.dumps({"docs": records}), encoding="utf-8")


def test_missing_documents_are_retained_by_default(registry_paths, monkeypatch):
    """Missing records stay available as historical entries unless opted in."""
    docs_path, _ = registry_paths
    _write_registry(docs_path, [
        {"id": "a", "status": "missing", "name": "Missing"},
        {"id": "b", "status": "failed", "name": "Failed"},
    ])
    monkeypatch.setattr(sys, "argv", ["cleanup_registry.py"])

    assert cleanup.main() == 0
    remaining = json.loads(docs_path.read_text(encoding="utf-8"))["docs"]
    assert [item["id"] for item in remaining] == ["a"]


def test_all_missing_flag_prunes_missing_records(registry_paths, monkeypatch):
    """The opt-in flag drops missing history and removes its index collection."""
    docs_path, _ = registry_paths
    _write_registry(docs_path, [{
        "id": "a", "status": "missing", "name": "Missing", "collection": "c_a"
    }])
    calls = []
    monkeypatch.setattr(cleanup, "drop_collection", lambda name: calls.append(name))
    monkeypatch.setattr(sys, "argv", ["cleanup_registry.py", "--all-missing"])

    assert cleanup.main() == 0
    assert calls == ["c_a"]
    assert json.loads(docs_path.read_text(encoding="utf-8"))["docs"] == []


def test_collection_deletion_failure_preserves_registry(registry_paths, monkeypatch):
    """A failed index deletion must not leave the registry pointing nowhere."""
    docs_path, uploads = registry_paths
    original = [{"id": "a", "status": "failed", "collection": "c_a"}]
    _write_registry(docs_path, original)
    orphan = uploads / "orphan.pdf"
    orphan.write_bytes(b"keep until index cleanup succeeds")

    def fail(_name):
        """Simulate a Chroma persistence error."""
        raise RuntimeError("Chroma unavailable")

    monkeypatch.setattr(cleanup, "drop_collection", fail)
    monkeypatch.setattr(sys, "argv", ["cleanup_registry.py"])

    assert cleanup.main() == 1
    assert json.loads(docs_path.read_text(encoding="utf-8"))["docs"] == original
    assert orphan.exists()


def test_dry_run_does_not_mutate_registry(registry_paths, monkeypatch):
    """Dry-run reports candidates without writing the registry."""
    docs_path, _ = registry_paths
    original = [{"id": "a", "status": "failed", "collection": "c_a"}]
    _write_registry(docs_path, original)
    before = docs_path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["cleanup_registry.py", "--dry-run"])

    assert cleanup.main() == 0
    assert docs_path.read_bytes() == before


def test_no_stale_records_without_upload_directory(tmp_path, monkeypatch):
    """A clean registry succeeds even before the uploads folder exists."""
    docs_path = tmp_path / "documents.json"
    _write_registry(docs_path, [{"id": "a", "status": "ready"}])
    monkeypatch.setattr(cleanup, "DOCS_PATH", docs_path)
    monkeypatch.setattr(cleanup, "UPLOADS_DIR", tmp_path / "not-created")
    monkeypatch.setattr(sys, "argv", ["cleanup_registry.py"])

    assert cleanup.main() == 0
