# One-time setup for TOF Analysis notebooks (Windows PowerShell).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "-> Creating virtual environment..."
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create venv. Install Python 3.9+ from python.org and ensure 'python' is on PATH."
    exit 1
}

$py = Join-Path $PWD ".venv\Scripts\python.exe"

Write-Host "-> Installing package (re-run after git pull)..."
& $py -m pip install --upgrade pip -q
& $py -m pip install -e . -q

Write-Host "-> Registering Jupyter kernel..."
& $py -m ipykernel install --user --name=tof-analysis --display-name="Python (TOF Analysis)"

Write-Host ""
Write-Host "Ready. In PowerShell:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  jupyter notebook notebooks\calibration.ipynb"
Write-Host ""
Write-Host "In Cursor / VS Code: open the project folder, pick interpreter"
Write-Host "  .venv\Scripts\python.exe"
Write-Host "  then notebook kernel 'Python (TOF Analysis)'"
