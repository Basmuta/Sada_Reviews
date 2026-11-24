@echo off
setlocal
pushd "%~dp0"

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 (
    py --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py"
)

if not defined PYTHON_CMD (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo Python was not found. Install Python 3.11 or newer, enable Add Python to PATH, then run this file again.
    echo Download: https://www.python.org/downloads/
    pause
    popd
    exit /b 1
)

%PYTHON_CMD% --version
%PYTHON_CMD% -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 15) else 1)"
if errorlevel 1 (
    echo This project supports Python 3.11 through 3.15. Install one of these versions and try again.
    pause
    popd
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto setup_error
    if not exist ".venv\Scripts\python.exe" goto setup_error
)

echo Installing or checking project packages. This can take several minutes on the first run...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_error

if not exist "models\sada_camelbert_finetuned\config.json" (
    echo.
    echo No fine-tuned CAMeLBERT checkpoint was found.
    echo The sample dashboard will open; review classification needs the model from the Colab notebook.
    echo.
)

echo Opening Sada Reviews at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run app.py
popd
exit /b %errorlevel%

:setup_error
echo.
echo Setup did not finish. Check that Python 3.11 through 3.15 and an internet connection are available, then try again.
pause
popd
exit /b 1

