@echo off
setlocal EnableExtensions
title _VOIS RAG One-Click Installer

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Project tools, packages, models, and caches will be kept with this folder.

set "BOOTSTRAP=%PROJECT_ROOT%\_rag_bootstrap.ps1"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$bat = [IO.Path]::GetFullPath('%~f0');" ^
  "$root = [IO.Path]::GetFullPath('%PROJECT_ROOT%');" ^
  "$raw = [IO.File]::ReadAllText($bat);" ^
  "$payload = ($raw -split '(?m)^# POWERSHELL_PAYLOAD\r?$', 2)[1].TrimStart();" ^
  "$ps = [IO.Path]::GetFullPath('%BOOTSTRAP%');" ^
  "[IO.File]::WriteAllText($ps, $payload, [Text.UTF8Encoding]::new($false));" ^
  "& powershell -NoProfile -ExecutionPolicy Bypass -File $ps -ProjectRoot $root;" ^
  "exit $LASTEXITCODE"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo Setup failed with exit code %EXITCODE%.
  exit /b %EXITCODE%
)

endlocal
exit /b 0

# POWERSHELL_PAYLOAD
param(
  [string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Log($Message) { Write-Host "`n[INFO] $Message" -ForegroundColor Cyan }
function Warn($Message) { Write-Host "`n[WARN] $Message" -ForegroundColor Yellow }
function Fail($Message) { Write-Host "`n[ERROR] $Message" -ForegroundColor Red; exit 1 }

function Download-File($Url, $OutFile) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
  $partial = "$OutFile.download"
  Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
  Log "Downloading $Url"
  try {
    Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
  } catch {
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
      & curl.exe -L --fail --retry 3 --output $partial $Url
      if ($LASTEXITCODE -ne 0) { throw "curl failed for $Url" }
    } else {
      throw
    }
  }
  if (-not (Test-Path -LiteralPath $partial) -or
      (Get-Item -LiteralPath $partial).Length -eq 0) {
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    throw "Download was empty or incomplete: $Url"
  }
  Move-Item -LiteralPath $partial -Destination $OutFile -Force
}

function Assert-MicrosoftSignature($FilePath) {
  $signature = Get-AuthenticodeSignature -LiteralPath $FilePath
  if ($signature.Status -ne 'Valid' -or
      -not $signature.SignerCertificate.Subject.Contains('Microsoft Corporation')) {
    throw "Authenticode verification failed for Microsoft installer: $FilePath"
  }
}

function Add-ProcessPath($PathToAdd) {
  if (($env:Path -split ';') -notcontains $PathToAdd) {
    $env:Path = "$PathToAdd;$env:Path"
  }
}

function Test-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $p = [Security.Principal.WindowsPrincipal]$id
  return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSCommandPath }
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path $ProjectRoot)) { New-Item -ItemType Directory -Force -Path $ProjectRoot | Out-Null }
$ToolsDir = Join-Path $ProjectRoot '.tools'
$DownloadsDir = Join-Path $ToolsDir 'downloads'
$PythonDir = Join-Path $ToolsDir 'Python311'
$GitDir = Join-Path $ToolsDir 'Git'
$OllamaDir = Join-Path $ToolsDir 'Ollama'
$ModelsDir = Join-Path $ToolsDir 'ollama-models'
$PipCache = Join-Path $ToolsDir 'pip-cache'
$TempDir = Join-Path $ToolsDir 'tmp'
$VenvDir = Join-Path $ProjectRoot '.venv-rag'
$ChromaDir = Join-Path $ProjectRoot 'chroma_db'
$AppFile = Join-Path $ProjectRoot 'Rag.py'

New-Item -ItemType Directory -Force -Path $ToolsDir,$DownloadsDir,$OllamaDir,$ModelsDir,$PipCache,$TempDir,$ChromaDir | Out-Null
$MainLog = Join-Path $ToolsDir 'setup.log'
try { Start-Transcript -LiteralPath $MainLog -Append | Out-Null } catch {}

$env:TEMP = $TempDir
$env:TMP = $TempDir
$env:PIP_CACHE_DIR = $PipCache
$env:OLLAMA_MODELS = $ModelsDir

Log "Project root: $ProjectRoot"
Log "All project-managed tools and caches will use: $ToolsDir"

try {
  Log 'Checking Visual C++ Redistributable'
  $vcInstalled = $false
  foreach ($key in @(
    'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64'
  )) {
    if (Test-Path $key) {
      $v = Get-ItemProperty $key -ErrorAction SilentlyContinue
      if ($v.Installed -eq 1) { $vcInstalled = $true }
    }
  }
  if (-not $vcInstalled) {
    $vc = Join-Path $DownloadsDir 'vc_redist.x64.exe'
    Download-File 'https://aka.ms/vs/17/release/vc_redist.x64.exe' $vc
    Assert-MicrosoftSignature $vc
    if (-not (Test-Admin)) { Warn 'VC++ Redistributable may require admin rights. Windows may prompt for elevation.' }
    $p = Start-Process -FilePath $vc -ArgumentList '/install','/quiet','/norestart' -Wait -PassThru
    if ($p.ExitCode -notin @(0, 1638, 3010)) { throw "VC++ Redistributable installer failed with exit code $($p.ExitCode)" }
  }

  Log 'Checking Python'
  $PythonExe = Join-Path $PythonDir 'python.exe'
  if (-not (Test-Path $PythonExe)) {
    $py = Join-Path $DownloadsDir 'python-3.11.9.zip'
    Download-File 'https://www.nuget.org/api/v2/package/python/3.11.9' $py
    $extract = Join-Path $TempDir 'python-nuget'
    if (Test-Path $extract) { Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue }
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    Expand-Archive -LiteralPath $py -DestinationPath $extract -Force
    $tools = Join-Path $extract 'tools'
    if (-not (Test-Path (Join-Path $tools 'python.exe'))) { throw 'Python NuGet package did not contain tools\python.exe.' }
    if (Test-Path $PythonDir) { Remove-Item -LiteralPath $PythonDir -Recurse -Force -ErrorAction SilentlyContinue }
    New-Item -ItemType Directory -Force -Path $PythonDir | Out-Null
    Copy-Item -Path (Join-Path $tools '*') -Destination $PythonDir -Recurse -Force
  }
  Add-ProcessPath $PythonDir
  Add-ProcessPath (Join-Path $PythonDir 'Scripts')
  & $PythonExe -m ensurepip --upgrade

  Log 'Creating virtual environment in the project folder'
  if (-not (Test-Path (Join-Path $VenvDir 'Scripts\python.exe'))) {
    & $PythonExe -m venv $VenvDir
  }
  $VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
  & $VenvPython -m pip install --upgrade pip setuptools wheel

  Log 'PyTorch is not required; inference runs through the local Ollama service.'

  Log 'Installing RAG Python packages to the project virtual environment'
  & $VenvPython -m pip install --upgrade -r (Join-Path $ProjectRoot 'requirements.txt')
  if ($LASTEXITCODE -ne 0) {
    Warn 'Package install failed. Installing Microsoft Build Tools and retrying once.'
    $bt = Join-Path $DownloadsDir 'vs_BuildTools.exe'
    Download-File 'https://aka.ms/vs/17/release/vs_BuildTools.exe' $bt
    Assert-MicrosoftSignature $bt
    $BuildToolsDir = Join-Path $ToolsDir 'VSBuildTools'
    $btArgs = @(
      '--quiet','--wait','--norestart','--nocache',
      '--installPath', $BuildToolsDir,
      '--add','Microsoft.VisualStudio.Workload.VCTools',
      '--includeRecommended'
    )
    if (-not (Test-Admin)) { Warn 'Build Tools may require admin rights. Windows may prompt for elevation.' }
    $p = Start-Process -FilePath $bt -ArgumentList $btArgs -Wait -PassThru
    if ($p.ExitCode -notin @(0, 3010)) { throw "Build Tools installer failed with exit code $($p.ExitCode)" }
    & $VenvPython -m pip install --upgrade -r (Join-Path $ProjectRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed after Build Tools retry.' }
  }

  Log 'Checking Ollama'
  $OllamaExe = Join-Path $OllamaDir 'ollama.exe'
  if (-not (Test-Path $OllamaExe)) {
    $assetUrl = $null
    try {
      $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/ollama/ollama/releases/latest' -UseBasicParsing
      $asset = $release.assets | Where-Object { $_.name -eq 'ollama-windows-amd64.zip' } | Select-Object -First 1
      if (-not $asset) {
        $asset = $release.assets | Where-Object { $_.name -match 'windows.*amd64.*\.zip$' -and $_.name -notmatch 'mlx' } | Select-Object -First 1
      }
      if ($asset) { $assetUrl = $asset.browser_download_url }
    } catch {}
    if ($assetUrl) {
      $zip = Join-Path $DownloadsDir 'ollama-windows-amd64.zip'
      Download-File $assetUrl $zip
      Expand-Archive -LiteralPath $zip -DestinationPath $OllamaDir -Force
      $found = Get-ChildItem -LiteralPath $OllamaDir -Filter 'ollama.exe' -Recurse | Select-Object -First 1
      if ($found -and $found.FullName -ne $OllamaExe) { Copy-Item -LiteralPath $found.FullName -Destination $OllamaExe -Force }
    }
    if (-not (Test-Path $OllamaExe)) {
      throw 'Portable Ollama ZIP was unavailable or invalid. Refusing to install under the Windows user profile; retry when the download is available.'
    }
  }
  Add-ProcessPath (Split-Path -Parent $OllamaExe)

  Log 'Starting Ollama service'
  $ollamaOutLog = Join-Path $ToolsDir 'ollama.out.log'
  $ollamaErrLog = Join-Path $ToolsDir 'ollama.err.log'
  $env:OLLAMA_HOST = '127.0.0.1:11435'
  $env:OLLAMA_BASE_URL = 'http://127.0.0.1:11435'
  $alreadyReady = $false
  try {
    Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/tags' -UseBasicParsing -TimeoutSec 3 | Out-Null
    $alreadyReady = $true
  } catch {}
  if (-not $alreadyReady) {
    Start-Process -FilePath $OllamaExe -ArgumentList 'serve' -WindowStyle Hidden -RedirectStandardOutput $ollamaOutLog -RedirectStandardError $ollamaErrLog
  }
  $ready = $false
  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
      Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/tags' -UseBasicParsing | Out-Null
      $ready = $true
      break
    } catch {}
  }
  if (-not $ready) { throw "Ollama service did not become ready. See $ollamaOutLog and $ollamaErrLog" }

  function Test-OllamaModel($ModelName) {
    & $OllamaExe show $ModelName *> $null
    return ($LASTEXITCODE -eq 0)
  }

  function Pull-OllamaModel($ModelName, $TimeoutMinutes) {
    if (Test-OllamaModel $ModelName) {
      Write-Host "$ModelName is already installed."
      return $true
    }
    $safe = ($ModelName -replace '[^a-zA-Z0-9._-]', '_')
    $pullOut = Join-Path $ToolsDir "ollama-pull-$safe.out.log"
    $pullErr = Join-Path $ToolsDir "ollama-pull-$safe.err.log"
    Write-Host "Pulling $ModelName ..."
    $p = Start-Process -FilePath $OllamaExe -ArgumentList @('pull', $ModelName) -RedirectStandardOutput $pullOut -RedirectStandardError $pullErr -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddMinutes($TimeoutMinutes)
    while (-not $p.HasExited) {
      Start-Sleep -Seconds 10
      if ((Get-Date) -gt $deadline) {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        Warn "Timed out pulling $ModelName after $TimeoutMinutes minutes. See $pullErr"
        return $false
      }
    }
    if ($p.ExitCode -eq 0 -or (Test-OllamaModel $ModelName)) { return $true }
    Warn "Failed to pull $ModelName with exit code $($p.ExitCode). See $pullErr"
    return $false
  }

  Log 'Pulling Ollama models into the project folder'
  $SelectedModel = 'qwen2.5:3b'
  if (-not (Pull-OllamaModel $SelectedModel 20)) {
    throw "Failed to pull required chat model: $SelectedModel"
  }
  if (-not (Pull-OllamaModel 'nomic-embed-text' 15)) {
    throw 'Failed to pull required embedding model: nomic-embed-text'
  }
  if (-not (Pull-OllamaModel 'bge-m3' 20)) {
    throw 'Failed to pull required Arabic embedding model: bge-m3'
  }
  $env:RAG_MODEL = $SelectedModel

  $AppFile = Join-Path $ProjectRoot 'Rag.py'
  if (-not (Test-Path $AppFile)) {
    throw "Project app not found: $AppFile"
  }
  Log 'Using the project application: Rag.py'

  Log 'Launching application'
  Write-Host ''
  Write-Host '============================================================' -ForegroundColor Green
  Write-Host 'RAG app is starting. Open this URL when Gradio is ready:' -ForegroundColor Green
  Write-Host 'http://127.0.0.1:7860' -ForegroundColor Green
  Write-Host '============================================================' -ForegroundColor Green
  Write-Host ''
  & $VenvPython $AppFile
} catch {
  Write-Host ''
  Write-Host 'Automatic setup failed:' -ForegroundColor Red
  Write-Host $_.Exception.Message -ForegroundColor Red
  Write-Host ''
  Write-Host "Logs and downloads are in: $ToolsDir" -ForegroundColor Yellow
  Write-Host "Main setup log: $MainLog" -ForegroundColor Yellow
  exit 1
}
