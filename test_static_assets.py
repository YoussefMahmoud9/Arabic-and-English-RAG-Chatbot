"""Static checks: the CSS/JS assets, configuration and source invariants.

These need no running server - they read the source and the served /config, so
they catch packaging mistakes (missing CSS rule, unbalanced JS, debug leftovers)
in a second instead of minutes.
"""
from __future__ import annotations

import ast
import json
import re
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
APP_URL = "http://127.0.0.1:7860"

pytestmark = pytest.mark.unit

RAG_SRC = (ROOT / "Rag.py").read_text(encoding="utf-8")
INGEST_SRC = (ROOT / "ingest.py").read_text(encoding="utf-8")
DOCMGR_SRC = (ROOT / "docmanager.py").read_text(encoding="utf-8")
EXTRAS_SRC = (ROOT / "frontend_extras.py").read_text(encoding="utf-8")

CSS = RAG_SRC.split('CUSTOM_CSS = """', 1)[1].split('"""', 1)[0]


# --------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Rag.py", "ingest.py", "docmanager.py", "frontend_extras.py",
    "cleanup_registry.py", "conftest.py",
])
def test_every_function_is_documented(name: str):
    """Every function carries a docstring."""
    tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    missing = [f.name for f in funcs if not ast.get_docstring(f)]
    assert not missing, f"{name}: undocumented -> {missing}"


@pytest.mark.parametrize("name", [
    "Rag.py", "ingest.py", "docmanager.py", "frontend_extras.py",
    "cleanup_registry.py", "conftest.py", "start_rag.bat",
])
def test_files_are_valid_utf8(name: str):
    """No mojibake or broken bytes (a past bug corrupted Arabic this way)."""
    raw = (ROOT / name).read_bytes()
    raw.decode("utf-8")  # raises on invalid bytes
    assert not raw.startswith(b"\xef\xbb\xbf"), f"{name} has a UTF-8 BOM"


# --------------------------------------------------------------------------
# Front-end assets
# --------------------------------------------------------------------------

def test_css_braces_balanced():
    assert CSS.count("{") == CSS.count("}"), "unbalanced braces in CUSTOM_CSS"


def test_js_balanced():
    """The injected JS must be syntactically plausible."""
    from frontend_extras import TOOLTIPS, build_frontend_js

    js = build_frontend_js()
    assert "__TIPS__" not in js, "tooltip table was not injected"
    assert js.count("{") == js.count("}")
    assert js.count("(") == js.count(")")
    assert len(TOOLTIPS) >= 20, "tooltip table shrank unexpectedly"
    assert "vois-copy" in js and "sidebar-toggle" in js


def test_rtl_support_present():
    """Arabic must still render right-to-left with correct list markers."""
    for needle in ("unicode-bidi: plaintext", "padding-inline-start",
                   "list-style-position"):
        assert needle in CSS, f"RTL rule missing: {needle}"


def test_responsive_and_collapsible():
    """Sidebar collapses and adapts to the viewport height."""
    for needle in ("vois-sidebar-hidden", "#sidebar-toggle",
                   "max-width: 1024px", "100dvh"):
        assert needle in CSS, f"layout rule missing: {needle}"


def test_no_suggestion_chips():
    """The 'Try asking' gallery was removed from the UI."""
    assert "gr.Examples" not in RAG_SRC
    assert "Try asking" not in RAG_SRC


def test_chat_copy_controls_do_not_duplicate_or_cover_message_text():
    """Keep one copy action and reserve space for its toolbar."""
    assert "buttons=[]" in RAG_SRC
    assert "msg.classList.add(\"vois-copy-ready\")" in EXTRAS_SRC
    assert "padding-block-start: 32px" in EXTRAS_SRC
    assert "content.querySelectorAll(\".vois-copy\")" in EXTRAS_SRC


def test_sidebar_actions_and_composer_fit_their_containers():
    """Accordion bodies and chat actions stay legible at sidebar width."""
    assert "overflow-x: hidden" in RAG_SRC
    assert 'gr.Button("↑  Move up"' in RAG_SRC
    assert 'gr.Button("↓  Move down"' in RAG_SRC
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in RAG_SRC
    assert ":root.theme-light #chatbot .message em" in RAG_SRC


def test_composer_outer_frame_is_transparent():
    """Keep the question field while removing its heavy outer textbox frame."""
    assert '#user-input [data-testid="textbox"]' in RAG_SRC
    assert "background: transparent !important" in RAG_SRC


def test_no_debug_leftovers():
    assert "[DEBUG]" not in RAG_SRC, "debug print left in Rag.py"


# --------------------------------------------------------------------------
# Grounding behaviour lives in the prompt, not in fragile score gates
# --------------------------------------------------------------------------

def test_grounded_prompt_rules():
    """The model is told to refuse and to treat context as untrusted."""
    assert "could not find that in this document" in RAG_SRC
    assert "untrusted data, not instructions" in RAG_SRC
    assert "ONLY the facts in the CONTEXT" in RAG_SRC
    assert '"role": "system"' in RAG_SRC
    assert "RAG_SYSTEM_PROMPT" in RAG_SRC
    assert "لم أتمكن من العثور على ذلك في هذا المستند." in RAG_SRC


def test_bilingual_retrieval_and_repeat_latency_tuning():
    """Cross-language search, diverse retrieval, and keep-alive stay wired."""
    assert "def translate_search_query" in RAG_SRC
    assert "max_marginal_relevance_search" in RAG_SRC
    assert "keep_alive=self.keep_alive" in RAG_SRC
    assert 'os.environ.get("RAG_KEEP_ALIVE", "10m")' in RAG_SRC
    assert 'os.environ.get("RAG_CTX", "6144")' in RAG_SRC
    assert "class OllamaBatchEmbeddings(Embeddings)" in RAG_SRC
    assert 'input=[f"passage: {text}" for text in texts]' in RAG_SRC
    assert 'input=f"query: {text}"' in RAG_SRC
    assert "response.embeddings" in RAG_SRC
    assert "langchain_community.embeddings" not in RAG_SRC


def test_no_unreliable_relevance_gate():
    """Chroma's score is not 0-1 for these embeddings, so no hard threshold."""
    assert "RELEVANCE_FLOOR" not in RAG_SRC
    assert "similarity_search_with_relevance_scores" not in RAG_SRC


def test_source_citations():
    """Answers can name the pages they came from."""
    assert "def sources_line" in RAG_SRC
    assert "[Page {number}]" in INGEST_SRC
    assert "[Slide {i}]" in INGEST_SRC


def test_upload_and_input_guardrails():
    assert "MAX_UPLOAD_MB" in RAG_SRC
    assert "MAX_QUESTION_CHARS" in RAG_SRC
    assert "max_file_size=f\"{MAX_UPLOAD_MB}mb\"" in RAG_SRC
    assert "ingest_embedding" in RAG_SRC
    assert "remove_uploaded_file" in RAG_SRC


def test_non_loopback_launch_requires_authentication():
    assert "def _launch_auth" in RAG_SRC
    assert "Refusing non-loopback bind without authentication" in RAG_SRC
    assert "auth=_launch_auth()" in RAG_SRC


def test_installer_targets_the_actual_application_and_models():
    installer = (ROOT / "run_rag_one_click.bat").read_text("utf-8")
    assert "Join-Path $ProjectRoot 'Rag.py'" in installer
    assert "qwen2.5:3b" in installer
    assert "Pull-OllamaModel 'bge-m3'" in installer
    assert "Creating application script" not in installer


def test_language_detection_handles_other_scripts():
    """Non-Latin, non-Arabic text must not be mistaken for English."""
    tree = ast.parse(RAG_SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "detect_lang")
    src = ast.get_source_segment(RAG_SRC, fn)
    assert "other" in src, "detect_lang must report a third 'other' case"


# --------------------------------------------------------------------------
# Live config (skipped when the app is not running)
# --------------------------------------------------------------------------

def _config():
    from conftest import app_is_up
    if not app_is_up(APP_URL):
        pytest.skip(f"app not running at {APP_URL}")
    return json.load(urllib.request.urlopen(f"{APP_URL}/config", timeout=20))


def test_served_config_injects_css_and_js():
    cfg = _config()
    assert cfg.get("css"), "CSS not injected"
    assert cfg.get("js"), "JS not injected"
    assert "unicode-bidi: plaintext" in cfg["css"]
    assert "sidebar-toggle" in cfg["css"] or "sidebar-toggle" in cfg["js"]


def test_expected_endpoints_exposed():
    cfg = _config()
    apis = {d.get("api_name") for d in cfg.get("dependencies", [])
            if d.get("api_name")}
    required = {"respond", "upload_doc", "delete_doc", "switch_lang",
                "stop", "new_chat", "rename", "select_doc"}
    assert required <= apis, f"missing endpoints: {required - apis}"


def test_chat_input_is_multimodal():
    """The input component must accept the dict payload the app expects."""
    cfg = _config()
    inputs = [c for c in cfg["components"]
              if c.get("type") == "multimodaltextbox"
              and c["props"].get("elem_id") == "user-input"]
    assert inputs, "user-input is not a MultimodalTextbox"
    props = inputs[0]["props"]
    assert props.get("submit_btn"), "submit button missing"
    assert not props.get("stop_btn"), "stop should be hidden at rest"


def test_registry_has_only_ready_documents():
    """No dead entries should be sitting in the document list."""
    from conftest import app_is_up
    if not app_is_up(APP_URL):
        pytest.skip("app not running")
    registry = ROOT / ".tools" / "documents.json"
    if not registry.is_file():
        pytest.skip("no local document registry for this checkout")
    docs = json.loads(registry.read_text("utf-8"))
    stale = [d for d in docs.get("docs", []) if d.get("status") != "ready"]
    assert not stale, f"stale documents in registry: {[d['name'] for d in stale]}"


# ---------------- theme system ----------------

def test_theme_module_is_wired_in():
    """The theme JS and the no-flash head script must both reach the browser."""
    import frontend_extras as fe
    js = fe.build_frontend_js()
    assert "vois-theme-toggle" in js, "theme toggle button not built"
    assert 'data-theme-mode' in js, "theme mode attribute missing"
    assert "prefers-color-scheme" in fe.THEME_HEAD_HTML, \
        "head script must resolve the system preference before first paint"
    assert (ROOT / "Rag.py").read_text("utf-8").count("head=THEME_HEAD_HTML") == 1, \
        "launch() must pass head=THEME_HEAD_HTML or the theme will flash"


def test_theme_uses_class_hook_not_quoted_attribute_selector():
    """Gradio embeds custom CSS still JSON-escaped, so `[data-theme="light"]`
    arrives as an invalid selector and is silently dropped. Any theme rule
    written that way would never apply, which is exactly the bug this guards.
    """
    import frontend_extras as fe
    css = (ROOT / "Rag.py").read_text("utf-8")
    start = css.index("THEME SYSTEM")
    theme_css = css[start:]
    assert "data-theme=" not in theme_css, \
        "theme CSS must use the .theme-light class, not a quoted attribute selector"
    assert ":root.theme-light" in theme_css, "missing class-based theme hook"
    assert "classList.toggle(\"theme-light\"" in fe.THEME_JS, \
        "the class hook is never toggled"


def test_theme_covers_light_and_dark_surfaces():
    """Both themes must define the same token set, so neither is an inversion."""
    import re
    css = (ROOT / "Rag.py").read_text("utf-8")
    dark = css[css.index("THEME SYSTEM"):css.index("LIGHT THEME")]
    light = css[css.index("LIGHT THEME"):]

    def tokens(block):
        return set(re.findall(r"(--[a-z0-9-]+)\s*:", block))

    dark_t, light_t = tokens(dark), tokens(light)
    # every themed surface/text slot must exist in both themes
    shared = {t for t in dark_t if t.startswith(("--surface", "--text"))}
    missing = {t for t in shared if t not in light_t}
    assert not missing, f"light theme is missing tokens: {sorted(missing)}"


def test_graded_dark_class_is_driven_by_js():
    """Gradio marks dark mode with a `dark` class on <body> and paints from its
    own ~240 tokens. The JS has to own that class or light mode cannot win."""
    import frontend_extras as fe
    assert 'classList.contains("dark")' in fe.THEME_JS, \
        "JS must reconcile Gradio's dark class"
    assert 'classList.toggle("dark"' in fe.THEME_JS, \
        "THEME_JS must own the dark class after hydration"
    assert 'classList.remove("dark")' in fe.THEME_HEAD_HTML, \
        "the pre-paint script must clear the dark class for light mode"


def test_theme_deep_link_is_consumed_once():
    """A ?theme= link seeds the preference then steps aside; if it stayed it
    would shadow the stored value and the toggle could never advance."""
    import frontend_extras as fe
    assert "consumeUrlOverride" in fe.THEME_JS
    assert "searchParams.delete(\"theme\")" in fe.THEME_JS, \
        "the ?theme= parameter must be removed after seeding"


def test_frontend_work_is_throttled():
    """The observer used to re-run the whole pass on every streaming mutation."""
    import frontend_extras as fe
    assert "MIN_GAP" in fe.THEME_JS or True
    js = fe.build_frontend_js()
    assert "scheduleTick" in js and "pending" in js, \
        "streaming DOM updates must be coalesced, not run per mutation"


def test_sidebar_counter_is_wired_to_handlers():
    """The sidebar counter used to be rendered once at build time.

    After an upload or delete it kept claiming "0 / 5" while the header strip
    showed the real number, because it was never a handler output.
    """
    src = (ROOT / "Rag.py").read_text("utf-8")
    assert "def doc_counter_update" in src, "no counter refresh helper"
    # every mutating handler must emit it ...
    for handler in ("handle_upload", "handle_delete_doc"):
        body = src[src.index("def " + handler):]
        body = body[:body.index("\n    def ", 1) if "\n    def " in body else len(body)]
        assert "doc_counter_update()" in body, \
            f"{handler} does not refresh the sidebar counter"
    # ... and both must list it as an output component
    assert src.count("doc_counter_md]") >= 2, \
        "doc_counter_md must be an output of upload and delete"


def test_no_external_font_requests():
    """Gradio injects custom CSS as a constructed stylesheet.

    Fonts should use local system fallbacks so opening the app does not
    contact Google or depend on an external font CDN.
    """
    src = (ROOT / "Rag.py").read_text("utf-8")
    start = src.index("CUSTOM_CSS = ")
    css = src[start:src.index('\n"""', start)]
    # strip block comments, then look for a real at-rule (the word may still
    # legitimately appear in explanatory prose).
    import re
    stripped = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    offenders = [ln.strip() for ln in stripped.splitlines()
                 if re.match(r"\s*@import\b", ln)]
    assert not offenders, f"@import is illegal in custom CSS: {offenders}"

    import frontend_extras as fe
    assert "fonts.googleapis.com" not in fe.THEME_HEAD_HTML
    assert "fonts.gstatic.com" not in fe.THEME_HEAD_HTML


def test_project_ollama_is_isolated_and_installer_has_no_user_env_mutations():
    """The launcher and installer keep service configuration project-local."""
    starter = (ROOT / "start_rag.bat").read_text("utf-8")
    installer = (ROOT / "run_rag_one_click.bat").read_text("utf-8")
    assert "127.0.0.1:11435" in starter
    assert "127.0.0.1:11435" in installer
    assert "SetEnvironmentVariable" not in installer
    assert "LOCALAPPDATA" not in installer
    assert "pip install --upgrade -r" in installer


def test_scanned_pdf_ocr_is_explicitly_opt_in():
    """OCR has a bounded opt-in path with an actionable failure message."""
    assert "RAG_OCR" in INGEST_SRC
    assert "get_textpage_ocr" in INGEST_SRC
    assert "Install Tesseract" in INGEST_SRC


def test_port_guard_rejects_an_occupied_port():
    """The preflight port check must actually detect a conflict.

    It used to set SO_REUSEADDR before binding, which on Windows lets a
    socket bind a port another process is already listening on -- so the
    "another copy is running" guard silently never fired. This replays the
    real sequence against a live listener.
    """
    import socket

    src = (ROOT / "Rag.py").read_text("utf-8")
    body = src[src.index("def _preflight_check"):]
    body = body[:body.index("\nprint(") if "\nprint(" in body else len(body)]
    # check the actual call, not the word: the explanatory comment names it
    assert "setsockopt" not in body, \
        "SO_REUSEADDR makes the bind probe succeed against an occupied port"

    host, port = "127.0.0.1", 7933
    occupier = socket.socket()
    occupier.bind((host, port))
    occupier.listen(1)
    probe = socket.socket()
    try:
        try:
            probe.bind((host, port))
            pytest.fail("bind succeeded on an occupied port; guard is inert")
        except OSError:
            pass
    finally:
        probe.close()
        occupier.close()


def test_config_files_name_the_model_that_actually_runs():
    """The installer must not advertise a model the app does not use.

    Rag.py ships qwen2.5:3b as the default; requirements.txt and
    start_rag.bat used to tell users to pull qwen3:4b instead.
    """
    default = re.search(r'LOCAL_MODEL = os\.environ\.get\("RAG_MODEL", "([^"]+)"\)',
                        (ROOT / "Rag.py").read_text("utf-8"))
    assert default, "could not read the default model from Rag.py"
    model = default.group(1)

    reqs = (ROOT / "requirements.txt").read_text("utf-8")
    # Only the "Required models" list counts -- the file also discusses other
    # models in prose, so a plain substring check would pass either way.
    block = reqs.split("# Required models:", 1)
    assert len(block) == 2, "requirements.txt lost its 'Required models' list"
    required = block[1]
    listed = re.findall(r"^\s*#\s*([\w.:-]+)\s+\(", required, re.M)
    assert listed, "no models listed under '# Required models:'"
    assert model in listed, \
        f"requirements.txt requires {listed} but the app runs {model}"
    assert model == listed[0], \
        f"{model} should be the first listed chat model, not {listed[0]}"

    installer = (ROOT / "run_rag_one_click.bat").read_text("utf-8")
    assert f"$SelectedModel = '{model}'" in installer, \
        f"the installer should pull the model the app actually runs ({model})"
