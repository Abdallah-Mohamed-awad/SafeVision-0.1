@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-api\Scripts\python.exe" (
  echo API virtual environment not found. Create it with: py -3.11 -m venv .venv-api
  pause
  exit /b 1
)
if not exist "web\index.html" (
  echo Missing web\index.html. Put the frontend file in the web folder.
  pause
  exit /b 1
)
if not exist "models\best.pt" (
  echo Missing models\best.pt. Copy the selected trained checkpoint there.
  pause
  exit /b 1
)
.venv-api\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
