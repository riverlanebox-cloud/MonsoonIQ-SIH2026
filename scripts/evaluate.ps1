# MonsoonIQ Evaluation & Verification Script
Write-Host "=== Running Verification on Test Years (2022-2023) and Compiling PDF Report ===" -ForegroundColor Cyan
$env:PYTHONPATH = "."
.venv\Scripts\python.exe src/evaluate.py
Write-Host "=== Evaluation Complete! Verification PDF and JSON metrics saved ===" -ForegroundColor Green
