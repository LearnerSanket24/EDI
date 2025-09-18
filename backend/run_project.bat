@echo off
echo Starting AI Exam Monitor Project...

REM Open the frontend in the default web browser
echo Opening frontend in web browser...
start "" "file:///C:/Users/sanke/OneDrive/Documents/EDI/Project/Project/frontend/index.html"

REM Wait a moment
timeout /t 2

REM Open API documentation in browser (main endpoint page)
echo Opening API endpoints page...
start "" "http://127.0.0.1:5000/"

echo.
echo Project components started:
echo - Backend Flask API running on http://127.0.0.1:5000
echo - Frontend opened in web browser
echo - API endpoints: http://127.0.0.1:5000/
echo - Head pose detection: http://127.0.0.1:5000/api/detections/head_pose
echo - Model debug info: http://127.0.0.1:5000/api/debug/model
echo.
echo The trained MobileNet head pose detection model is loaded and ready!
echo.
pause
