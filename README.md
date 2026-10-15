# Arabic and English RAG Chatbot

## Run on Windows

1. Place the project folder on a drive with enough free space and run `run_rag_one_click.bat` once to install the project-managed tools, Python packages, and Ollama models.
2. Run `start_rag.bat` to start the app.
3. Open `http://127.0.0.1:7860` after Gradio reports that it is running.

The project-managed Ollama service uses `127.0.0.1:11435` and stores model data under `.tools\ollama-models`. This avoids accidentally using a separate Ollama installation on its usual port. Python packages, the virtual environment, caches, Chroma index, and application data are stored in the project folder. The Visual C++ runtime is a Windows system component and may be installed system-wide; it cannot be made fully portable by placing it beside the project.

See the [user and technical guide](docs/VOIS_RAG_User_and_Technical_Guide.pdf) and the [system overview presentation](docs/VOIS_RAG_System_Overview.pptx) for setup steps, architecture, retrieval flow, and the conceptual data model.

## Speed And Language Quality

The app keeps the chat model loaded for 10 minutes after use to reduce reload delays. It uses a bounded context size, the five best diverse evidence chunks, and at most four recent history turns. Arabic questions against English-only indexes are translated for search only; the original Arabic question remains the question sent to the answer model. The first such cross-language query can take longer, and its translation is cached for repeat queries.

Before starting the app, these optional environment variables can tune memory and retrieval: `RAG_KEEP_ALIVE` (for example `0` to unload immediately), `RAG_CTX` (default `6144`), `RAG_TOP_K` (default `5`), and `RAG_RETRIEVAL_DIVERSITY` (default `0.75`, range `0` to `1`). Higher diversity can return broader evidence; lower diversity favors near-duplicate top matches.

## Scanned PDFs

Text extraction is the default. OCR is optional because it requires a local Tesseract installation and can be substantially slower. Install Tesseract with the needed language data, ensure `tesseract.exe` is on `PATH` (and `TESSDATA_PREFIX` is set if needed), then set these variables in the same terminal before launching `start_rag.bat`:

```powershell
$env:RAG_OCR = "1"
$env:RAG_OCR_LANG = "eng+ara"
.\start_rag.bat
```

Use Tesseract language codes present in your installed `tessdata` directory. For example, use `eng` if Arabic data is not installed. OCR runs only for PDF pages with little or no extractable text. If OCR is disabled, scanned PDFs receive an actionable error message instead of silently indexing empty content.

## Security And Data

- The app binds to loopback by default. Do not expose it to a network without setting both `RAG_USERNAME` and `RAG_PASSWORD` and putting external access behind TLS.
- Document text is untrusted input. System-role instructions and explicit context boundaries reduce prompt-injection risk, but local language models cannot guarantee that injected instructions will always be ignored. Check source excerpts for important answers.
- Uploaded files, chat history, settings, and the Chroma index are local files and are not encrypted by the application. Protect the Windows account and project directory; use BitLocker for at-rest protection where available.
- `cleanup_registry.py --dry-run` previews cleanup. By default, entries marked `missing` are retained. `--all-missing` opts into removing them.
