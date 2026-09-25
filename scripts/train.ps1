# MonsoonIQ Model Training Script
Write-Host "=== Training MonsoonIQ Regime Classifier, MoE Experts & Heavy Rain Modules ===" -ForegroundColor Cyan
$env:PYTHONPATH = "."
.venv\Scripts\python.exe src/train.py
Write-Host "=== Training Complete! Artifacts saved to artifacts/models/ ===" -ForegroundColor Green
