# MonsoonIQ Backend API Server Launcher
Write-Host "=== Starting MonsoonIQ FastAPI Backend on http://127.0.0.1:8000 ===" -ForegroundColor Cyan
$env:PYTHONPATH = "."
.venv\Scripts\uvicorn.exe src.api.main:app --host 127.0.0.1 --port 8000 --reload
