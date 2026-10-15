"""Front-end enhancements injected into the Gradio page at load time.

Two jobs:
  1. Attach a hover tooltip to every control that does not already have one,
     so the purpose of each button is discoverable.
  2. Add a "copy" affordance to each assistant message.

Kept as one self-contained string so the Python side stays readable.
"""

TOOLTIPS = {
    # --- chat input area -------------------------------------------------
    "user-input": "Type your question and press Enter to send",
    # --- translation / language ------------------------------------------
    "tr-btn": "Switch language (English <-> Arabic) and translate this answer",
    # --- documents -------------------------------------------------------
    "doc-dd": "Pick the document to answer from",
    "bind-doc-box": "Open a document by typing its exact name",
    "bind-doc-btn": "Open the document named above",
    "upload-btn": "Upload a PDF, DOCX or PPTX to add to the library",
    "doc-rename-box": "New name for the current document",
    "doc-rename-btn": "Rename the current document",
    "doc-del-btn": "Delete the current document and its index",
    "doc-counter-wrap": "How many documents are loaded (maximum 5)",
    "doc-banner": "Warnings about the current document appear here",
    "active-doc-line": "The document answers are currently taken from",
    # --- chats -----------------------------------------------------------
    "newchat-btn": "Start a fresh conversation",
    "history-dd": "Open a previous conversation",
    "chat-list": "All conversations for this document - click one to open it",
    "move-up": "Move this chat up the list",
    "move-down": "Move this chat down the list",
    "chat-del-btn": "Delete this conversation",
    "rename-box": "New title for this conversation",
    "rename-btn": "Rename this conversation",
    # --- misc ------------------------------------------------------------
    "chatbot": "Conversation with the document",
    "brand": "Application header",
    "model-badge": "The local model answering your questions",
    "footer-note": "Answers come from the selected document only",
}

# The tooltip widget is plain CSS driven by a data-tip attribute, so it needs
# no extra DOM beyond adding the attribute itself.
TOOLTIP_JS = r"""
(function () {
  const TIPS = __TIPS__;

  function tooltipCss() {
    if (document.getElementById("vois-tooltip-style")) return;
    const style = document.createElement("style");
    style.id = "vois-tooltip-style";
    style.textContent = `
      [data-tip] { position: relative; }
      [data-tip]:hover::after {
        content: attr(data-tip);
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%) translateY(4px);
        background: #0b1226;
        color: #e6ecff;
        border: 1px solid #3d4d70;
        border-radius: 8px;
        padding: 7px 11px;
        font-size: 12.5px;
        line-height: 1.45;
        font-weight: 500;
        white-space: normal;
        width: max-content;
        max-width: 260px;
        text-align: center;
        box-shadow: 0 10px 28px rgba(0,0,0,.55);
        opacity: 0;
        pointer-events: none;
        transition: opacity .16s ease, transform .16s ease;
        z-index: 9999;
      }
      [data-tip]:hover::after { opacity: 1; transform: translateX(-50%) translateY(0); }
      @media (prefers-reduced-motion: reduce) {
        [data-tip]:hover::after { transition: none; }
      }
    `;
    document.head.appendChild(style);
  }

  function applyTips() {
    tooltipCss();
    for (const id of Object.keys(TIPS)) {
      const el = document.getElementById(id);
      if (el && !el.getAttribute("data-tip")) {
        el.setAttribute("data-tip", TIPS[id]);
      }
    }
  }

  // --- one copy action per chat message --------------------------------
  function addCopyButtons() {
    document.querySelectorAll("#chatbot .message").forEach((msg) => {
      if (msg.getAttribute("data-copy-ready")) return;
      if (!msg.textContent || !msg.textContent.trim()) return;

      msg.setAttribute("data-copy-ready", "1");
      msg.classList.add("vois-copy-ready");
      msg.style.position = "relative";

      const btn = document.createElement("button");
      btn.className = "vois-copy";
      btn.type = "button";
      btn.textContent = "Copy";
      btn.setAttribute("aria-label", "Copy this answer");
      btn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        try {
          const content = msg.cloneNode(true);
          content.querySelectorAll(".vois-copy").forEach((el) => el.remove());
          await navigator.clipboard.writeText(content.innerText.trim());
          btn.textContent = "Copied";
          btn.classList.add("done");
        } catch (e) {
          btn.textContent = "Failed";
        }
        setTimeout(() => {
          btn.textContent = "Copy";
          btn.classList.remove("done");
        }, 1600);
      });
      msg.appendChild(btn);
    });
  }

  function copyCss() {
    if (document.getElementById("vois-copy-style")) return;
    const style = document.createElement("style");
    style.id = "vois-copy-style";
    style.textContent = `
      .vois-copy {
        position: absolute;
        top: 8px;
        inset-inline-end: 8px;
        opacity: 0;
        pointer-events: none;
        background: rgba(20, 184, 183, .16);
        color: #7fe6e5;
        border: 1px solid rgba(20, 184, 183, .45);
        border-radius: 8px;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0;
        padding: 4px 9px;
        cursor: pointer;
        transition: opacity .16s ease, transform .16s ease, background .16s ease;
      }
      #chatbot .message:hover .vois-copy,
      #chatbot .message:focus-within .vois-copy,
      .vois-copy:focus-visible { opacity: 1; pointer-events: auto; }
      .vois-copy:hover { background: rgba(20, 184, 183, .3); transform: translateY(-1px); }
      .vois-copy.done { background: rgba(34, 197, 94, .28); color: #bbf7d0;
                        border-color: rgba(34, 197, 94, .6); }
      #chatbot .vois-copy-ready { padding-block-start: 32px !important; }
    `;
    document.head.appendChild(style);
  }

  function tick() { copyCss(); applyTips(); addCopyButtons(); }

  // --- collapsible sidebar -------------------------------------------
  // Runs entirely in the browser so the panel opens instantly and costs
  // no server round-trip. The state is remembered between visits and the
  // sidebar is forced shut on narrow screens where it would crowd the chat.
  const STORE_KEY = "vois.sidebarHidden";
  const NARROW = 1024;      // must match the CSS breakpoint

  function sidebarEl() { return document.getElementById("sidebar"); }

  function isNarrow() { return window.innerWidth <= NARROW; }

  function isHidden() {
    return document.body.classList.contains("vois-sidebar-hidden");
  }

  function placeToggle() {
    const btn = document.getElementById("sidebar-toggle");
    const sb = sidebarEl();
    if (!btn || !sb) return;
    if (isHidden() || isNarrow()) {
      btn.style.left = "10px";
    } else {
      // Sit just inside the panel's right edge.
      const w = sb.getBoundingClientRect().width;
      btn.style.left = Math.max(10, w - 44) + "px";
    }
  }

  // Collapse the panel by walking up from #sidebar to the layout row and
  // hiding every wrapper in between. Gradio nests columns in extra elements,
  // so styling #sidebar alone leaves the outer wrapper still holding its
  // flex basis and the chat never grows. `display: none` is used because it
  // takes the whole subtree out of layout, which is the only reliable way to
  // reclaim the space; a short opacity fade softens the jump.
  const COLLAPSE_PROPS = ["display", "flex", "width", "min-width", "max-width",
                          "padding-left", "padding-right", "overflow"];

  function setCollapsed(hidden) {
    const sb = sidebarEl();
    if (!sb) return;
    let node = sb;
    while (node && node.id !== "app-shell") {
      if (hidden) {
        node.style.setProperty("display", "none", "important");
      } else {
        COLLAPSE_PROPS.forEach((p) => node.style.removeProperty(p));
      }
      node = node.parentElement;
    }
    const shell = document.getElementById("app-shell");
    if (shell) shell.style.setProperty("gap", hidden ? "0" : "", "important");
  }

  // Tracks whether the *current* hidden state came from the automatic
  // narrow-screen collapse (as opposed to the user or a ?sidebar= param).
  // Without this, the auto-restore below would fight the user's own choice.
  function markAuto(on) {
    document.body.dataset.voisAutoHidden = on ? "1" : "0";
  }

  function applyState(hidden, persist, auto) {
    document.body.classList.toggle("vois-sidebar-hidden", hidden);
    document.body.classList.toggle("vois-drawer-open", !hidden && isNarrow());
    markAuto(!!auto);
    setCollapsed(hidden && !isNarrow());
    const btn = document.getElementById("sidebar-toggle");
    if (btn) {
      btn.textContent = hidden ? "☰" : "✕";
      btn.setAttribute("data-tip", hidden
        ? "Show the sidebar (Ctrl+B)"
        : "Hide the sidebar (Ctrl+B)");
      btn.setAttribute("aria-label", hidden ? "Show sidebar" : "Hide sidebar");
      btn.setAttribute("aria-expanded", String(!hidden));
    }
    if (persist) {
      try { localStorage.setItem(STORE_KEY, hidden ? "1" : "0"); } catch (e) {}
    }
    placeToggle();
  }

  function buildToggle() {
    if (document.getElementById("sidebar-toggle")) return;
    const btn = document.createElement("button");
    btn.id = "sidebar-toggle";
    btn.type = "button";
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      applyState(!isHidden(), true, false);   // manual: never auto-restored
    });
    document.body.appendChild(btn);
  }

  function initSidebar() {
    const sb = sidebarEl();
    if (!sb) return;          // Gradio has not rendered it yet; retried below
    buildToggle();

    // Decide the starting state exactly once, so later re-runs cannot undo
    // a choice the user has just made with the button.
    if (!window.__voisSidebarReady) {
      // A ?sidebar=hidden|shown query wins over the saved preference, which
      // makes a full-screen chat view linkable (and testable).
      let forced = null;
      try {
        const q = new URLSearchParams(window.location.search).get("sidebar");
        if (q === "hidden" || q === "collapsed") forced = true;
        else if (q === "shown" || q === "open") forced = false;
      } catch (e) {}

      let stored = null;
      try { stored = localStorage.getItem(STORE_KEY); } catch (e) {}

      if (forced !== null) {
        applyState(forced, false, false);
      } else if (isNarrow()) {
        // Too tight to show both: start collapsed, and remember that this
        // was automatic so widening the window can restore it.
        applyState(true, false, true);
      } else {
        applyState(stored === "1", false, false);
      }
      window.__voisSidebarReady = true;
    } else {
      // Already initialised: re-assert the state, because the first pass may
      // have run before Gradio had rendered #sidebar.
      setCollapsed(isHidden() && !isNarrow());
      const btn = document.getElementById("sidebar-toggle");
      if (btn) {
        btn.textContent = isHidden() ? "☰" : "✕";
        btn.setAttribute("data-tip", isHidden()
          ? "Show the sidebar (Ctrl+B)"
          : "Hide the sidebar (Ctrl+B)");
        btn.setAttribute("aria-label", isHidden() ? "Show sidebar" : "Hide sidebar");
        btn.setAttribute("aria-expanded", String(!isHidden()));
      }
      placeToggle();
    }

    // Keyboard shortcuts.
    if (!window.__voisSidebarKeys) {
      window.__voisSidebarKeys = true;
      document.addEventListener("keydown", (ev) => {
        var mod = ev.ctrlKey || ev.metaKey;
        if (mod && (ev.key === "b" || ev.key === "B")) {
          ev.preventDefault();
          applyState(!isHidden(), true, false);
        } else if (mod && (ev.key === "k" || ev.key === "K")) {
          // Jump to the composer: the fastest path back into the chat.
          var box = document.querySelector("#user-input textarea");
          if (box) {
            ev.preventDefault();
            box.focus();
            var n = box.value.length;
            try { box.setSelectionRange(n, n); } catch (e) {}
          }
        } else if (ev.key === "Escape" && !isHidden() && isNarrow()) {
          applyState(true, true, false);
        }
      });
    }

    // Clicking the drawer backdrop closes it.
    if (!window.__voisBackdrop) {
      window.__voisBackdrop = true;
      document.addEventListener("click", (ev) => {
        if (!document.body.classList.contains("vois-drawer-open")) return;
        if (ev.target === document.body) applyState(true, true, false);
      }, true);
    }

    if (!window.__voisResizeBound) {
      window.__voisResizeBound = true;
      window.addEventListener("resize", () => {
        if (isNarrow() && !isHidden()) {
          // Crossing into narrow: collapse so the chat keeps its width.
          applyState(true, false, true);
        } else if (!isNarrow() && isHidden() &&
                   document.body.dataset.voisAutoHidden === "1") {
          // Crossing back to wide: only undo a collapse WE caused.
          applyState(false, false, false);
        }
        placeToggle();
      });
    }

    // The auto/manual flag is set explicitly inside applyState(), so there
    // is deliberately no observer here: observing the class would mark a
    // deliberate hide as "automatic" and let the resize handler undo it.
  }

  // --- throttled work queue ---------------------------------------------
  // Gradio mutates the DOM continuously while a response streams. Running the
  // full pass on every mutation meant rewriting two <style> elements and
  // re-scanning every message dozens of times per second, which showed up as
  // jank on long conversations. Coalesce the bursts into at most one pass
  // every MIN_GAP ms, with a trailing pass so the last change still lands.
  var MIN_GAP = 120;
  var lastRun = 0;
  var pending = false;
  var rafId = 0;

  function scheduleTick() {
    var now = new Date().getTime();
    var wait = Math.max(0, MIN_GAP - (now - lastRun));
    if (wait === 0) { runTick(); return; }
    if (pending) return;
    pending = true;
    rafId = window.setTimeout(function () {
      pending = false;
      runTick();
    }, wait);
  }

  function runTick() {
    lastRun = new Date().getTime();
    tick();
    initSidebar();
  }

  function start() {
    runTick();
    const obs = new MutationObserver(scheduleTick);
    obs.observe(document.body, { childList: true, subtree: true, characterData: true });
    document.addEventListener("gradio:load", runTick);
    window.addEventListener("load", runTick);
    // Belt and braces: a few timed retries in case no mutation is observed.
    [200, 600, 1200, 2500, 4000].forEach(function (ms) {
      window.setTimeout(runTick, ms);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
"""


def build_frontend_js() -> str:
    """Return the JavaScript blob handed to Gradio's launch(js=...).

    The tooltip table is injected as JSON so the text stays editable in
    Python and needs no manual escaping inside the JavaScript.
    """
    import json
    tips = json.dumps(TOOLTIPS, ensure_ascii=False)
    return TOOLTIP_JS.replace("__TIPS__", tips) + THEME_JS


THEME_HEAD_HTML = """
<script>
/* Applies the saved theme before first paint so switching themes never
   flashes the wrong background. Mirrors the logic in THEME_JS; keep the
   storage key ("vois-theme") and values ("light"/"dark"/"system") in sync. */
(function () {
  try {
    var q = new URLSearchParams(window.location.search).get("theme");
    var saved = (q === "light" || q === "dark" || q === "system")
      ? q
      : localStorage.getItem("vois-theme");
    var sysDark = window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    var t = (saved === "light" || saved === "dark")
      ? saved
      : (saved === "system" || !saved ? (sysDark ? "dark" : "light") : saved);
    var root = document.documentElement;
    root.setAttribute("data-theme", t);
    // A class is toggled too: Gradio embeds custom CSS still JSON-escaped, so
    // attribute selectors with quotes are invalid in the served stylesheet.
    // Classes need no quoting and therefore survive.
    root.classList.toggle("theme-light", t === "light");
    // Gradio paints from its own token set and marks dark mode with a `dark`
    // class on <body>, which flips ~240 variables. Mirror the choice onto that
    // class so the framework's own components follow the theme instead of
    // being fought with per-selector overrides.
    if (t === "light" && document.body) {
      document.body.classList.remove("dark");
    }
  } catch (e) {}
})();
</script>
"""


THEME_JS = r"""
/* ---------------- theme switching (light / dark / system) --------------
   The CSS exposes a [data-theme="light"] override layer; this module owns
   the state, the toggle button and the system-preference listener. Kept in
   its own block so the existing tooltip/copy logic stays untouched.
   ------------------------------------------------------------------- */
(function () {
  var KEY = "vois-theme";
  var ORDER = ["light", "dark", "system"];
  var ICON = { light: "\u2600", dark: "\u263D", system: "\u25D0" };
  var NAME = { light: "Light", dark: "Dark", system: "System" };
  var mq = null;
  try {
    mq = window.matchMedia("(prefers-color-scheme: dark)");
  } catch (e) {}

  function getMode() {
    try {
      var v = localStorage.getItem(KEY);
      return (v === "light" || v === "dark" || v === "system") ? v : "system";
    } catch (e) { return "system"; }
  }

  // A ?theme= deep link seeds the initial preference and is then removed from
  // the URL. Left in place it would shadow the stored value on every read, so
  // the toggle button could never advance past the first step.
  function consumeUrlOverride() {
    try {
      var q = new URLSearchParams(window.location.search).get("theme");
      if (q !== "light" && q !== "dark" && q !== "system") return;
      localStorage.setItem(KEY, q);
      var url = new URL(window.location.href);
      url.searchParams.delete("theme");
      window.history.replaceState({}, "", url.toString());
    } catch (e) {}
  }

  function resolved(mode) {
    if (mode === "light" || mode === "dark") return mode;
    return (mq && mq.matches) ? "dark" : "light";
  }

  function syncGradioClass(resolvedMode) {
    // Gradio re-applies its own theme during hydration, so re-assert our
    // choice a few times right after load instead of trusting one pass.
    if (!document.body) return;
    var wantDark = resolvedMode === "dark";
    if (document.body.classList.contains("dark") !== wantDark) {
      document.body.classList.toggle("dark", wantDark);
    }
  }

  function apply(mode, persist) {
    if (ORDER.indexOf(mode) === -1) mode = "system";
    var resolvedMode = resolved(mode);
    var root = document.documentElement;
    root.setAttribute("data-theme", resolvedMode);
    root.setAttribute("data-theme-mode", mode);
    // Class-based hook: Gradio ships custom CSS still JSON-escaped, so a
    // quoted attribute selector never matches. Classes survive intact.
    root.classList.toggle("theme-light", resolvedMode === "light");
    syncGradioClass(resolvedMode);
    var btn = document.getElementById("vois-theme-toggle");
    if (btn) {
      btn.textContent = ICON[mode];
      btn.setAttribute("data-tip", "Theme: " + NAME[mode] +
        (mode === "system" ? " (follows your device)" : "") +
        " -- click to change");
      btn.setAttribute("aria-label", "Theme: " + NAME[mode] +
        ". Activate to change.");
    }
    if (persist) {
      try { localStorage.setItem(KEY, mode); } catch (e) {}
    }
  }

  function build() {
    if (document.getElementById("vois-theme-toggle")) return;
    var btn = document.createElement("button");
    btn.id = "vois-theme-toggle";
    btn.type = "button";
    btn.className = "vois-theme-btn";
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      var next = ORDER[(ORDER.indexOf(getMode()) + 1) % ORDER.length];
      apply(next, true);
    });
    document.body.appendChild(btn);
  }

  function init() {
    build();
    consumeUrlOverride();
    apply(getMode(), false);
    // Re-assert for a few seconds: Gradio sets its `dark` class during
    // hydration and can land after our first pass.
    var tries = 0;
    var iv = window.setInterval(function () {
      syncGradioClass(resolved(getMode()));
      if (++tries >= 10 && iv) window.clearInterval(iv);
    }, 400);
  }

  // Ctrl/Cmd+Shift+T cycles the theme without leaving the keyboard.
  if (!window.__voisThemeKeys) {
    window.__voisThemeKeys = true;
    document.addEventListener("keydown", function (ev) {
      if (!(ev.ctrlKey || ev.metaKey) || !ev.shiftKey) return;
      if (ev.key !== "T" && ev.key !== "t") return;
      ev.preventDefault();
      apply(ORDER[(ORDER.indexOf(getMode()) + 1) % ORDER.length], true);
    });
  }

  // Follow the OS while in "system" mode, without clobbering an explicit pick.
  if (mq && mq.addEventListener) {
    mq.addEventListener("change", function () {
      if (getMode() === "system") apply("system", false);
    });
  } else if (mq && mq.addListener) {
    mq.addListener(function () {
      if (getMode() === "system") apply("system", false);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  window.addEventListener("gradio:load", init);
  window.addEventListener("load", init);
  window.setTimeout(init, 300);
})();
"""
