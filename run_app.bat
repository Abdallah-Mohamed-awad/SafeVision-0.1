@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)

if not exist "app.py" (
  echo app.py is missing from the project root.
  pause
  exit /b 1
)

if not exist "models\best.pt" (
  if exist "models\candidates\yolo26s_best.pt" (
    echo Using the trained YOLO26s candidate as best.pt...
    copy /Y "models\candidates\yolo26s_best.pt" "models\best.pt" >nul
  ) else if exist "models\legacy_baseline_best.pt" (
    echo Using the legacy trained checkpoint as best.pt...
    copy /Y "models\legacy_baseline_best.pt" "models\best.pt" >nul
  ) else if exist "models\candidates\yolo26n_best.pt" (
    echo Using the trained YOLO26n candidate as best.pt...
    copy /Y "models\candidates\yolo26n_best.pt" "models\best.pt" >nul
  ) else (
    echo No trained checkpoint found in models\.
    pause
    exit /b 1
  )
)

.venv\Scripts\python.exe -m streamlit run app.py
