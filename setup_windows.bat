@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 was not found. Install it, then reopen PowerShell.
  pause
  exit /b 1
)

echo [2/4] Creating or reusing virtual environment...
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo Existing .venv detected; reusing it.
)

if not exist "requirements.txt" (
  echo requirements.txt is missing from the project root.
  goto :error
)

echo [3/4] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [4/4] Installing project requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup completed successfully.
echo Next: run run_app.bat
pause
exit /b 0

:error
echo.
echo Setup failed. Read the error above.
pause
exit /b 1
