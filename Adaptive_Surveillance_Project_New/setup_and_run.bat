@echo off
title Adaptive Surveillance - VNR VJIET
color 0A

echo ============================================================
echo   Adaptive Object-Based Video Summarization
echo   VNR Vignana Jyothi Institute of Engineering and Technology
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Install Python 3.10 from https://www.python.org/downloads/
    echo Check "Add Python to PATH" during install!
    pause
    exit /b
)

echo [OK] Python found
echo.

:: Create venv if not exists
if not exist "venv\" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

:: Activate venv
call venv\Scripts\activate.bat

echo.
echo [SETUP] Installing packages (this takes 2-3 mins on first run)...
echo.

:: Install packages one by one to avoid timeout
pip install --quiet opencv-python
pip install --quiet numpy
pip install --quiet pandas
pip install --quiet moviepy
pip install --quiet pyyaml pillow requests psutil
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --quiet ultralytics --no-deps

echo.
echo [OK] All packages installed!
echo.

:: Create folders
if not exist "events"        mkdir events
if not exist "metadata"      mkdir metadata
if not exist "final_summary" mkdir final_summary
if not exist "models"        mkdir models

echo [OK] Folders ready
echo.
echo ============================================================
echo   Starting Surveillance System...
echo   Press ESC in camera window to stop.
echo ============================================================
echo.

python main.py

echo.
echo ============================================================
echo   Surveillance stopped. Generating final summary video...
echo ============================================================
echo.

python merge_events.py

echo.
echo [DONE] Check final_summary folder for summary video.
echo [DONE] Check metadata folder for events_metadata.csv
echo.
pause
