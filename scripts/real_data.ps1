# MonsoonIQ - build the REAL archive and run the whole chain on it (Windows PowerShell)
#   IMD 0.25 deg observed rainfall + archived NWP Day 1-5 forecasts + dynamics + OLR + terrain
# Every download is cached under data\real\raw, so re-running resumes where it stopped.
Write-Host "=== MonsoonIQ real-data pipeline ===" -ForegroundColor Cyan
$env:PYTHONPATH = "."

Write-Host "`n[1/4] Fetching IMD rainfall, NWP forecasts, dynamics, OLR, terrain ..." -ForegroundColor Yellow
.venv\Scripts\python.exe scripts\fetch_real_data.py all
if ($LASTEXITCODE -eq 2) { Write-Host "Open-Meteo daily quota reached - run this script again tomorrow to resume." -ForegroundColor Red; exit 2 }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:MONSOONIQ_PROFILE = "real"
Write-Host "`n[2/4] Training on real seasons ..." -ForegroundColor Yellow
.venv\Scripts\python.exe src\train.py
Write-Host "`n[3/4] Verifying on held-out real seasons ..." -ForegroundColor Yellow
.venv\Scripts\python.exe src\evaluate.py
Write-Host "`n[4/4] Building console timeline ..." -ForegroundColor Yellow
.venv\Scripts\python.exe -m src.console.artifacts

Write-Host "`nDone. Serve it with:  `$env:MONSOONIQ_PROFILE='real'; .venv\Scripts\uvicorn.exe src.api.main:app --port 8000" -ForegroundColor Green
