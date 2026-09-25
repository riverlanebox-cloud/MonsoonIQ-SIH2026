# MonsoonIQ Setup Script for Windows PowerShell
Write-Host "=== Setting up MonsoonIQ Environment ===" -ForegroundColor Cyan

# 1. Create and configure Python Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Install Frontend Node modules
Write-Host "Installing Frontend dependencies..." -ForegroundColor Yellow
Set-Location frontend
npm.cmd install
Set-Location ..

Write-Host "=== MonsoonIQ Environment Setup Complete! ===" -ForegroundColor Green
