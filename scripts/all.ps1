# MonsoonIQ End-to-End Orchestrator Script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "      MONSOONIQ: END-TO-END PIPELINE ORCHESTRATION        " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "."

# 1. Dataset Generation
Write-Host "`n[Step 1/4] Generating 8-Year Physically Plausible Dataset..." -ForegroundColor Yellow
.venv\Scripts\python.exe -m src.data.synthetic_generator

# 2. Model Training
Write-Host "`n[Step 2/4] Training Regime Classifier, MoE Experts, Quantile Regressors, and Heavy Rain Classifiers..." -ForegroundColor Yellow
.venv\Scripts\python.exe src/train.py

# 3. Model Evaluation & PDF Generation
Write-Host "`n[Step 3/4] Evaluating Held-Out Test Data & Compiling Publication PDF..." -ForegroundColor Yellow
.venv\Scripts\python.exe src/evaluate.py

# 4. Run Pytest Verification Suite
Write-Host "`n[Step 4/4] Executing Comprehensive Automated Test Suite..." -ForegroundColor Yellow
.venv\Scripts\pytest.exe tests/ -v

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "   MONSOONIQ FULL PIPELINE EXECUTION SUCCESSFUL!           " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "To start the backend API:  powershell ./scripts/api.ps1"
Write-Host "To start the frontend UI:  powershell ./scripts/ui.ps1"
