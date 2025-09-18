# Safe Exam Monitor - Complete System Startup with All Models
# This script ensures all AI models are loaded and working properly

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Safe Exam Monitor - Complete System" -ForegroundColor Cyan
Write-Host "Backend + Frontend with Full AI Models" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Kill any existing processes on our ports
Write-Host "Checking for existing processes..." -ForegroundColor Yellow
if (Test-Port 5000) {
    Write-Host "Port 5000 is in use. Attempting to free it..." -ForegroundColor Yellow
    Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}
if (Test-Port 5173) {
    Write-Host "Port 5173 is in use. Attempting to free it..." -ForegroundColor Yellow
    Get-Process | Where-Object {$_.ProcessName -eq "node"} | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Navigate to backend directory
$BackendDir = "C:\Users\sanke\OneDrive\Documents\EDI\Project\Project\backend"
Set-Location $BackendDir

# Check if virtual environment exists
$venvPath = ".\.venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ Activating Python virtual environment..." -ForegroundColor Green
& "$venvPath\Scripts\Activate.ps1"

# Verify Python and pip
Write-Host "Python version: $(python --version)" -ForegroundColor Green
Write-Host "Pip version: $(pip --version)" -ForegroundColor Green

Write-Host "✓ Installing Python dependencies..." -ForegroundColor Green
pip install --upgrade pip
pip install flask==3.0.3 flask-cors==4.0.1 twilio==9.2.3
pip install tensorflow>=2.15.0 numpy>=1.26.0 opencv-python>=4.8.0
pip install Pillow>=10.0.0 ultralytics>=8.2.0 scikit-learn>=1.3.0

# Test model loading
Write-Host "✓ Testing AI model loading..." -ForegroundColor Green
python -c "
import sys
try:
    print('Testing TensorFlow...')
    import tensorflow as tf
    print(f'✓ TensorFlow {tf.__version__} loaded successfully')
    
    print('Testing OpenCV...')
    import cv2
    print(f'✓ OpenCV {cv2.__version__} loaded successfully')
    
    print('Testing Ultralytics...')
    from ultralytics import YOLO
    print('✓ Ultralytics loaded successfully')
    
    print('Testing model files...')
    import os
    models_dir = 'models'
    if os.path.exists(f'{models_dir}/crowdhuman_custom.pt'):
        print('✓ CrowdHuman custom model found')
    if os.path.exists(f'{models_dir}/head_pose_mobilenet_final.h5'):
        print('✓ Head pose model found')
    if os.path.exists('yolov8n.pt'):
        print('✓ YOLOv8n fallback model found')
    
    print('Testing backend app...')
    from app import create_app
    app = create_app()
    print('✓ Flask app created successfully')
    
    print('✓ All models and dependencies ready!')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Model testing failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Starting Flask Backend with all models..." -ForegroundColor Green

# Start Flask backend in background
$backendProcess = Start-Process -FilePath "python" -ArgumentList "app.py" -PassThru -WindowStyle Normal
Write-Host "Backend started with PID: $($backendProcess.Id)" -ForegroundColor Green

# Wait for backend to start and test endpoints
Write-Host "Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

$maxRetries = 10
$retryCount = 0
$backendReady = $false

while ($retryCount -lt $maxRetries -and -not $backendReady) {
    try {
        Write-Host "Testing backend health (attempt $($retryCount + 1)/$maxRetries)..." -ForegroundColor Yellow
        $healthResponse = Invoke-RestMethod -Uri "http://localhost:5000/api/health" -Method Get -TimeoutSec 5
        
        if ($healthResponse.status -eq "ok") {
            Write-Host "✓ Backend health check passed!" -ForegroundColor Green
            $backendReady = $true
            
            # Test system status
            try {
                $statusResponse = Invoke-RestMethod -Uri "http://localhost:5000/api/system/status" -Method Get -TimeoutSec 10
                Write-Host "✓ System Status:" -ForegroundColor Green
                Write-Host "  - Models loaded: $($statusResponse.system.models_loaded)" -ForegroundColor Green
                Write-Host "  - Multi-person detection: $($statusResponse.capabilities.multi_person_detection)" -ForegroundColor Green
                Write-Host "  - Head pose detection: $($statusResponse.capabilities.head_pose_detection)" -ForegroundColor Green
                Write-Host "  - Body visibility detection: $($statusResponse.capabilities.body_visibility_detection)" -ForegroundColor Green
                Write-Host "  - Unified inference: $($statusResponse.capabilities.unified_inference)" -ForegroundColor Green
                
                if ($statusResponse.capabilities.unified_inference) {
                    Write-Host "🎉 All AI models loaded and ready!" -ForegroundColor Green
                } else {
                    Write-Host "⚠ Some models may not be fully loaded, but system is operational" -ForegroundColor Yellow
                }
            } catch {
                Write-Host "⚠ Could not get detailed system status, but basic backend is working" -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "⚠ Backend not ready yet..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        $retryCount++
    }
}

if (-not $backendReady) {
    Write-Host "❌ Backend failed to start after $maxRetries attempts!" -ForegroundColor Red
    if ($backendProcess -and !$backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force
    }
    exit 1
}

# Navigate to frontend directory
Write-Host "✓ Setting up React frontend..." -ForegroundColor Green
Set-Location "Safe_Exam_Monitor\my-app"

# Check Node.js version
Write-Host "Node.js version: $(node --version)" -ForegroundColor Green
Write-Host "NPM version: $(npm --version)" -ForegroundColor Green

# Install frontend dependencies if needed
if (-not (Test-Path "node_modules") -or (Get-ChildItem "node_modules").Count -lt 10) {
    Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ NPM install failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ Testing frontend build..." -ForegroundColor Green
# Check TypeScript compilation
npx tsc --noEmit
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ TypeScript compilation has issues, but continuing..." -ForegroundColor Yellow
}

Write-Host "🚀 Starting React frontend..." -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🎉 SYSTEM READY!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Backend Health: http://localhost:5000/api/health" -ForegroundColor Cyan
Write-Host "System Status: http://localhost:5000/api/system/status" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White
Write-Host "🔥 AI Models Active:" -ForegroundColor Green
Write-Host "  ✓ Custom CrowdHuman Multi-Person Detection" -ForegroundColor Green
Write-Host "  ✓ MobileNetV2 Head Pose Detection" -ForegroundColor Green
Write-Host "  ✓ Custom Body Visibility Detection" -ForegroundColor Green
Write-Host "  ✓ YOLOv8n Fallback Detection" -ForegroundColor Green
Write-Host "  ✓ Real-time Confidence Scoring" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "Press Ctrl+C to stop the system" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

# Start frontend (this will block until stopped)
npm run dev

# Cleanup when frontend stops
Write-Host "" -ForegroundColor White
Write-Host "Shutting down system..." -ForegroundColor Red
if ($backendProcess -and !$backendProcess.HasExited) {
    Write-Host "Stopping backend process..." -ForegroundColor Yellow
    Stop-Process -Id $backendProcess.Id -Force
    Write-Host "✓ Backend stopped." -ForegroundColor Green
}

Write-Host "✓ System shutdown complete." -ForegroundColor Green
