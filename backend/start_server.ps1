# Set environment variables for the head pose model
$env:HEAD_POSE_MODEL_PATH = "C:\Users\sanke\OneDrive\Documents\EDI\Project\Project\ml\models\head_pose_mobilenet_final.h5"
$env:HEAD_POSE_MODEL_INFO_PATH = "C:\Users\sanke\OneDrive\Documents\EDI\Project\Project\ml\models\head_pose_mobilenet_info.json"

# Activate virtual environment if not already active
if (-not $env:VIRTUAL_ENV) {
    & ".\.venv\Scripts\Activate.ps1"
}

# Print environment variables for verification
Write-Host "Model Path: $env:HEAD_POSE_MODEL_PATH"
Write-Host "Model Info Path: $env:HEAD_POSE_MODEL_INFO_PATH"

# Start the Flask server
python app.py
