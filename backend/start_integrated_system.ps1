# Start Integrated Safe Exam Monitor System
# This script starts both the Python backend and React frontend

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Safe Exam Monitor - Integrated System" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Check if Python virtual environment exists
$venvPath = ".\.venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "Activating Python virtual environment..." -ForegroundColor Green
& "$venvPath\Scripts\Activate.ps1"

Set-Location "backend"
Write-Host "Installing/Updating Python dependencies..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "Starting Python Flask Backend..." -ForegroundColor Green

# Start Flask backend in background
$backendProcess = Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory "." -PassThru -WindowStyle Minimized

Write-Host "Backend started with PID: $($backendProcess.Id)" -ForegroundColor Green

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Check if backend is healthy
try {
    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/health" -Method Get -TimeoutSec 10
    if ($response.status -eq "ok") {
        Write-Host "✓ Backend health check passed" -ForegroundColor Green
        
        # Get system status
        $status = Invoke-RestMethod -Uri "http://localhost:5000/api/system/status" -Method Get -TimeoutSec 10
        Write-Host "✓ Models loaded: $($status.system.models_loaded)" -ForegroundColor Green
        Write-Host "✓ Unified inference: $($status.capabilities.unified_inference)" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Backend health check failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Backend may still be starting..." -ForegroundColor Yellow
}

# Serve static frontend
Write-Host "Serving static frontend..." -ForegroundColor Green
Write-Host "Frontend will be available at: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Backend API available at: http://localhost:5000/api" -ForegroundColor Cyan

# Keep the script running to prevent immediate exit (frontend served by Flask)
Write-Host "System is running. Press Ctrl+C to stop." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Cleanup on exit
Write-Host "Shutting down..." -ForegroundColor Red
if ($backendProcess -and !$backendProcess.HasExited) {
    Write-Host "Stopping backend process..." -ForegroundColor Yellow
    Stop-Process -Id $backendProcess.Id -Force
    Write-Host "Backend stopped." -ForegroundColor Green
}

Write-Host "System shutdown complete." -ForegroundColor Green
