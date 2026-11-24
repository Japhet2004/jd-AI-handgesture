@echo off
title Hand Gesture Control for Windows
echo.
echo  ============================================
echo    Hand Gesture Control for Windows
echo    Starting...
echo  ============================================
echo.

cd /d "%~dp0"

REM Use existing venv or create one
if exist "venv\Scripts\python.exe" (
    echo  [OK] Virtual environment found.
) else (
    echo  [..] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo  [!!] Failed to create venv. Make sure Python 3 is installed.
        pause
        exit /b 1
    )
    echo  [..] Installing dependencies...
    venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo  [!!] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo  [OK] Setup complete!
)

echo.
echo  Launching hand gesture control...
echo  (Press Q or Esc in the camera window to quit)
echo.
venv\Scripts\python.exe hand_gesture_control.py

echo.
echo  Session ended.
pause
