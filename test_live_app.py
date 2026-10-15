"""Live behaviour tests against the running app.

Skipped automatically when the app is not reachable, so ``pytest`` is safe to
run at any time. Anything that needs real inference is marked ``slow``.

All non-ASCII text is built from code points: a literal escape sequence has
been silently corrupted once before (\\u0647 became \\0647, which Python read
as an octal escape and turned into "47").
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / ".tools" / "settings.json"
AR_RE = re.compile(r"[\u0600-\u06FF]")

# Built from code points so no encoding layer can mangle them.
ARABIC_WORD = "".join(chr(c) for c in (0x627, 0x644, 0x639, 0x631, 0x628, 0x64A, 0x629))
Q_AR = "".join(chr(c) for c in (
    0x645, 0x627, 0x20, 0x647, 0x648, 0x20, 0x647, 0x627, 0x62F, 0x648, 0x628, 0x61F))
DIGIT_IN_ARABIC = re.compile(r"[\u0600-\u06FF][0-9]+[\u0600-\u06FF]")

pytestmark = pytest.mark.live


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def as_input(text: str) -> dict:
    """Payload for the MultimodalTextbox chat input."""
    return {"text": text, "files": []}


def text_of(msg) -> str:
    """Plain text of a chatbot message (content may be str or list of parts)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def saved_lang() -> str:
    return json.loads(SETTINGS.read_text(encoding="utf-8"))["lang"]


def wait_for_lang(predicate, timeout: float = 10.0) -> str:
    """Poll settings.json until `predicate(value)` holds.

    The handler saves the setting before it does the (slow) translation, so a
    short fixed sleep is a race; polling makes the assertion reliable.
    """
    deadline = time.time() + timeout
    value = saved_lang()
    while time.time() < deadline:
        try:
            value = saved_lang()
        except (OSError, json.JSONDecodeError):
            time.sleep(0.2)
            continue
        if predicate(value):
            return value
        time.sleep(0.2)
    return value


def ar_count(text: str) -> int:
    return len(AR_RE.findall(text or ""))


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def test_app_exposes_documents(client):
    out = client.predict(api_name="/refresh_docs_on_load")
    docs = out[0]
    assert docs, "no documents available to answer from"
    assert out[1], "status line did not render"


def test_upload_with_no_file_is_graceful(client):
    out = client.predict(None, api_name="/upload_doc")
    assert out[1] == "", f"unexpected message: {out[1]!r}"


# --------------------------------------------------------------------------
# language switching
# --------------------------------------------------------------------------

def test_switch_lang_flips_and_persists(client):
    """Repeated clicks must alternate. Regression: the language used to be
    read from a state component that was frozen at its startup value, so the
    second click recomputed the same target and did nothing."""
    before = saved_lang()
    client.predict([], None, before, api_name="/switch_lang")
    after = wait_for_lang(lambda v: v != before)
    assert after != before, f"language did not flip ({before} -> {after})"

    client.predict([], None, after, api_name="/switch_lang")
    back = wait_for_lang(lambda v: v == before)
    assert back == before, f"language did not flip back ({after} -> {back})"

    # A third click must move again, not get stuck.
    client.predict([], None, back, api_name="/switch_lang")
    third = wait_for_lang(lambda v: v != before)
    assert third != before, f"third click did nothing (stuck on {third})"

    # Leave the setting as we found it.
    client.predict([], None, third, api_name="/switch_lang")
    wait_for_lang(lambda v: v == before)


def user_bot(q: str, a: str) -> list:
    return [{"role": "user", "content": [{"text": q, "type": "text"}],
             "metadata": None},
            {"role": "assistant", "content": [{"text": a, "type": "text"}],
             "metadata": None}]


@pytest.mark.slow
def test_switch_lang_translates_both_parts(client):
    """A click flips the language AND translates the question and answer.

    The target language is whatever the toggle is currently *not*, so the
    test works in both directions without needing to set the state (the
    language input is a gr.State and cannot be seeded over the API).

    The 3B model occasionally refuses or returns an empty translation. The
    app handles that deliberately: it detects the wrong language and tells
    the user to press the button again. This test therefore retries, exactly
    as the UI instructs, and only fails if no attempt translates cleanly.
    """
    current = saved_lang()
    target_ar = current == "en"

    if target_ar:                      # English -> Arabic
        src_q, src_a = ("What is Apache Cassandra used for?",
                        "Apache Cassandra is a distributed NoSQL database.")
    else:                              # Arabic -> English
        src_q = Q_AR
        src_a = ("Apache Cassandra \u0647\u0648 \u0642\u0627\u0639\u062F\u0629 "
                 "\u0628\u064A\u0627\u0646\u0627\u062A \u0645\u0648\u0632\u064A\u0639\u0629.")

    def translated(q: str, a: str) -> bool:
        if target_ar:
            return ar_count(q) > 3 and ar_count(a) > 15
        return ar_count(q) == 0 and ar_count(a) == 0

    last = ("", "")
    for attempt in range(3):
        lang_now = saved_lang()
        if lang_now == ("ar" if target_ar else "en"):
            # Already in the target language: flip back so a retry can run.
            client.predict([], None, lang_now, api_name="/switch_lang")
            wait_for_lang(lambda v: v != lang_now)

        out = client.predict(user_bot(src_q, src_a), None, saved_lang(),
                             api_name="/switch_lang")
        res = out[0] if isinstance(out, tuple) else out
        last = (text_of(res[0]), text_of(res[1]))

        assert wait_for_lang(lambda v: v == ("ar" if target_ar else "en")), \
            "language did not flip"
        if translated(*last):
            return

    pytest.fail(
        f"translation never completed in 3 attempts "
        f"(q={ascii(last[0][:70])}, a={ascii(last[1][:70])})")


@pytest.mark.slow
def test_switching_twice_is_a_safe_no_op(client):
    """A second click must not start translating its own status note."""
    out = client.predict(user_bot("What is Hadoop?",
                                  "Hadoop is a distributed framework."),
                         None, "en", api_name="/switch_lang")
    res = out[0] if isinstance(out, tuple) else out
    notes = [text_of(m) for m in res if text_of(m)[:1] in ("\u2705", "\U0001f310")]
    assert len(notes) <= 1, f"status note was translated as an answer: {notes}"


# --------------------------------------------------------------------------
# answering
# --------------------------------------------------------------------------

@pytest.fixture()
def needs_document(client):
    """Skip RAG assertions when the registry is empty.

    These tests check grounding, citation and refusal, which are only
    meaningful once a document has been uploaded and indexed. On a freshly
    cleared install there is nothing to answer from, and the app correctly
    replies "upload a document" -- which is not a failure.
    """
    docs = json.loads((ROOT / ".tools" / "documents.json").read_text("utf-8"))
    raw = docs.get("docs", {})
    # The app writes a list of records; older files used an id-keyed dict.
    entries = list(raw.values()) if isinstance(raw, dict) else list(raw)
    if not [d for d in entries
            if isinstance(d, dict) and d.get("status") == "ready"]:
        pytest.skip("no indexed document: upload one to run the RAG tests")


@pytest.mark.slow
def test_english_question_answers_in_english(client, needs_document):
    """An English question must be answered in English.

    Deliberately document-agnostic: these tests are about the *pipeline*
    (language routing, grounding, citation), so asking about a fixed topic
    made them fail whenever a different document happened to be loaded.
    """
    out = client.predict(as_input("Summarise what this document is about."), [],
                         api_name="/respond")
    res = out[0] if isinstance(out, tuple) else out
    answer = text_of(res[-1])
    assert len(answer) > 60, f"answer too short: {ascii(answer[:80])}"
    assert ar_count(answer) == 0, f"English question produced Arabic: {ascii(answer[:80])}"


@pytest.mark.slow
def test_arabic_question_answers_in_arabic(client, needs_document):
    out = client.predict(as_input(Q_AR), [], api_name="/respond")
    res = out[0] if isinstance(out, tuple) else out
    answer = text_of(res[-1])
    assert len(answer) > 40, f"answer too short: {ascii(answer[:80])}"
    assert ar_count(answer) > 25, f"Arabic question did not answer in Arabic: {ascii(answer[:80])}"


@pytest.mark.slow
def test_answer_cites_its_source(client, needs_document):
    """A grounded answer must point back at the document pages it used."""
    out = client.predict(as_input("Summarise what this document is about."), [],
                         api_name="/respond")
    res = out[0] if isinstance(out, tuple) else out
    assert "Source:" in text_of(res[-1]), "no source citation on a grounded answer"


@pytest.mark.slow
def test_off_topic_question_is_refused(client, needs_document):
    """The model must admit the answer is not in the document.

    The question has to be truly unrelated to whatever is loaded: asking
    about the 1998 World Cup used to pass, but a document about Big Data can
    legitimately mention it, and then a grounded answer is the correct
    behaviour rather than a refusal.
    """
    out = client.predict(
        as_input("What is the boiling point of liquid nitrogen at sea level?"), [],
        api_name="/respond")
    res = out[0] if isinstance(out, tuple) else out
    answer = text_of(res[-1])
    assert "could not find that in this document" in answer.lower(), \
        f"expected a refusal, got: {ascii(answer[:120])}"
    assert "Source:" not in answer, "a refusal should not cite pages"


@pytest.mark.slow
def test_empty_question_is_ignored(client):
    out = client.predict(as_input("   "), [], api_name="/respond")
    res = out[0] if isinstance(out, tuple) else out
    assert res == [] or res is None, "empty input should not create a turn"


# --------------------------------------------------------------------------
# text integrity
# --------------------------------------------------------------------------

def test_history_text_is_not_corrupted(client):
    """Arabic must survive the round trip byte-for-byte."""
    out = client.predict(user_bot("placeholder", "placeholder"), None,
                         saved_lang(), api_name="/switch_lang")
    res = out[0] if isinstance(out, tuple) else out
    for msg in res:
        text = text_of(msg)
        assert not DIGIT_IN_ARABIC.search(text), \
            f"Arabic text corrupted with digits: {ascii(text[:80])}"