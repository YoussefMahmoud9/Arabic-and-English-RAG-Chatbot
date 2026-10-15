@echo off
setlocal EnableExtensions
title _VOIS RAG Bot

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

set "TOOLS_DIR=%PROJECT_ROOT%\.tools"
set "VENV_PYTHON=%PROJECT_ROOT%\.venv-rag\Scripts\python.exe"
set "APP_FILE=%PROJECT_ROOT%\Rag.py"
set "MODELS_DIR=%TOOLS_DIR%\ollama-models"
set "OLLAMA_HOST=127.0.0.1:11435"
set "OLLAMA_BASE_URL=http://127.0.0.1:11435"

REM Keep models and caches inside the project directory.
set "OLLAMA_MODELS=%MODELS_DIR%"
set "PIP_CACHE_DIR=%TOOLS_DIR%\pip-cache"
set "TEMP=%TOOLS_DIR%\tmp"
set "TMP=%TOOLS_DIR%\tmp"
if not exist "%TEMP%" mkdir "%TOOLS_DIR%\tmp" >nul 2>&1

if not exist "%VENV_PYTHON%" (
  echo [ERROR] Virtual environment not found: %VENV_PYTHON%
  echo [ERROR] Run run_rag_one_click.bat first to set everything up.
  pause
  exit /b 1
)

if not exist "%APP_FILE%" (
  echo [ERROR] Application not found: %APP_FILE%
  pause
  exit /b 1
)

REM Use the project-local server so another installation cannot redirect models.
if not exist "%TOOLS_DIR%\Ollama\ollama.exe" (
  echo [ERROR] Project-managed Ollama is missing: %TOOLS_DIR%\Ollama\ollama.exe
  echo [ERROR] Run run_rag_one_click.bat to install it on this drive.
  pause
  exit /b 1
)
curl.exe --silent --fail "http://127.0.0.1:11435/api/tags" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Starting project Ollama ^(models stored with this project^)...
  start "" /min "%TOOLS_DIR%\Ollama\ollama.exe" serve 1>>"%TOOLS_DIR%\ollama.out.log" 2>>"%TOOLS_DIR%\ollama.err.log"
  set "OLLAMA_READY="
  for /L %%I in (1,1,30) do (
    timeout /t 2 /nobreak >nul
    curl.exe --silent --fail "http://127.0.0.1:11435/api/tags" >nul 2>&1 && set "OLLAMA_READY=1"
    if defined OLLAMA_READY goto ollama_ready
  )
  echo [ERROR] Project Ollama did not start on 127.0.0.1:11435.
  echo [ERROR] Check %TOOLS_DIR%\ollama.err.log or run ollama.exe serve manually.
  pause
  exit /b 1
)
:ollama_ready

echo.
echo ============================================================
echo   _VOIS RAG Bot is starting.
echo   Open when Gradio reports ready: http://127.0.0.1:7860
echo   Project Ollama: %OLLAMA_BASE_URL%
echo   Models stored with the project at %MODELS_DIR%
echo ============================================================
echo.

"%VENV_PYTHON%" "%APP_FILE%"

echo.
echo [INFO] The RAG app has stopped.
pause
