@echo off
echo ==========================================
echo Safe Exam Monitor - Starting System
echo ==========================================

cd /d "C:\Users\sanke\OneDrive\Documents\EDI\Project\Project\backend"

echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo Starting backend in new window...
start "Backend Server" /min python app.py

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo Starting frontend...
cd "Safe_Exam_Monitor\my-app"
echo.
echo ==========================================
echo System URLs:
echo Frontend: http://localhost:5173
echo Backend: http://localhost:5000
echo ==========================================
echo.

npm run dev

pause
