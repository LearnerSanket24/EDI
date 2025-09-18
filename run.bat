@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo === Safe Exam Monitor: Setup and Run ===

REM Change to the directory of this script
cd /d %~dp0

REM Check Python availability
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found in PATH. Install Python 3.10+ (64-bit) and retry.
  pause
  exit /b 1
)

REM Create virtual environment if missing
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv || (echo Failed to create venv & pause & exit /b 1)
)

echo Upgrading pip tooling...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

echo Installing minimal server dependencies...
".venv\Scripts\python.exe" -m pip install flask==3.0.3 flask-cors==4.0.1 Pillow>=10.0.0 numpy>=1.26.0 opencv-python>=4.8.0 scikit-learn>=1.3.0

echo Attempting TensorFlow install (for H5 head pose model)...
call ".venv\Scripts\python.exe" -m pip install tensorflow==2.17.0 || call ".venv\Scripts\python.exe" -m pip install tensorflow==2.16.1 || call ".venv\Scripts\python.exe" -m pip install tensorflow==2.15.0

echo Attempting Ultralytics install (for YOLO/CrowdHuman)...
call ".venv\Scripts\python.exe" -m pip install "ultralytics>=8.2.0"

echo Setting model paths...
set "HEAD_POSE_MODEL_PATH=%~dp0backend\models\head_pose_mobilenet_final.h5"
set "HEAD_POSE_MODEL_INFO_PATH=%~dp0backend\models\head_pose_mobilenet_info.json"

if exist "%HEAD_POSE_MODEL_PATH%" (
  echo ✓ Found head pose model: %HEAD_POSE_MODEL_PATH%
) else (
  echo ✗ Head pose model missing at %HEAD_POSE_MODEL_PATH%
)

if exist "%~dp0backend\models\crowdhuman_custom.pt" (
  echo ✓ Found CrowdHuman model: %~dp0backend\models\crowdhuman_custom.pt%
) else (
  echo ✗ CrowdHuman model missing in backend\models\
)

set "FLASK_ENV=development"
set "PORT=5000"

echo Starting backend on http://localhost:%PORT% ...
start "Backend" cmd /k ".venv\Scripts\python.exe" backend\app.py

echo Opening health and UI pages...
start "" http://localhost:%PORT%/api/health
start "" http://localhost:%PORT%/

echo If the page does not load, check the Backend window for errors and allow firewall access if prompted.
pause
endlocal
