# MonsoonIQ Dataset Generation Script
Write-Host "=== Generating MonsoonIQ Physically-Plausible Dataset (2016-2023) ===" -ForegroundColor Cyan
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m src.data.synthetic_generator
.venv\Scripts\python.exe -c "import pandas as pd; from src.data.validator import DataValidator; df = pd.read_parquet('data/synthetic/district_daily.parquet'); rep = DataValidator().validate_dataset_summary(df); print('Validation status:', rep['overall_valid'])"
Write-Host "=== Dataset Ready! ===" -ForegroundColor Green
