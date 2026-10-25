@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ZivaWhite -- starting...

where py >nul 2>nul
if %errorlevel%==0 (
    set PYCMD=py
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set PYCMD=python
    ) else (
        echo.
        echo Python was not found. Install Python 3.10+ from https://python.org/downloads/
        echo ^(tick "Add python.exe to PATH" during install^), then double-click this file again.
        echo.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating a virtual environment and installing dependencies...
    echo This happens once and can take a few minutes.
    %PYCMD% -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo.
        echo Failed to create a virtual environment. See any error above.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
    ".venv\Scripts\python.exe" -m pip install -e . --quiet
)

".venv\Scripts\python.exe" launch.py

echo.
echo ZivaWhite stopped. Press any key to close this window.
pause >nul
