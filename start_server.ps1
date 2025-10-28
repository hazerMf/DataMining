Set-Location -Path $PSScriptRoot\BE

# Check if virtual environment exists
if (-Not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Error: Virtual environment not found" -ForegroundColor Red
    exit 1
}

# Check if run_server.py exists
if (-Not (Test-Path "run_server.py")) {
    Write-Host "Error: run_server.py not found" -ForegroundColor Red
    exit 1
}

# Run server
.venv\Scripts\python.exe run_server.py
