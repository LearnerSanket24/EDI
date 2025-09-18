## AI-Powered Real-Time Exam Monitoring System

### Project Synopsis
Online examinations are widespread, but maintaining academic integrity at scale is challenging. This project proposes an AI-powered real-time exam monitoring system using computer vision and deep learning to automate invigilation, improving fairness and transparency while remaining affordable and privacy-conscious.

### Review of Literature
- CNNs for head pose detection
- YOLO-family detectors for person detection
- JavaScript Visibility API for tab-switch tracking
- Existing tools (e.g., ProctorU, Mettl) are proprietary; this project emphasizes open tech and transparency.

### Problem Statement and Objectives
- Robust, cost-effective, scalable real-time malpractice detection is lacking.
- Objectives:
  - Detect head movements (down/sideways)
  - Identify multiple people/background presence
  - Check upper-body visibility
  - Detect tab switching
  - Send real-time alerts via WhatsApp (Twilio)
  - Deliver a secure, user-friendly web app

### System Architecture
1. Web App: Frontend (HTML/CSS/JS) and Backend (Flask REST)
2. Webcam Monitoring: WebRTC capture each ~2s
3. AI Detection: Head pose (MobileNetV2), multi-person (YOLOv5), body visibility (landmarks), tab visibility
4. Alerting: Twilio WhatsApp
5. Storage: PostgreSQL/MongoDB for students, violations, logs

### Current Implementation (MVP)
- Flask backend with REST endpoints and CORS
- Mock AI endpoints: `/api/detections/*`
- WhatsApp alert placeholder endpoint `/api/alert`
- Simple browser client capturing webcam frames every 2s and sending to backend
- Tab visibility detection with immediate alert

### Repository Structure
```
Project/
  backend/
    app.py
    config.py
    alerts/
      twilio_client.py
    detections/
      head_pose.py
      multi_person.py
      body_visibility.py
    requirements.txt
  frontend/
    index.html
    script.js
```

### Getting Started
1) Backend
```
cd Project/backend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
setx FLASK_ENV development
setx PORT 5000
python app.py
```

2) Frontend
Open `Project/frontend/index.html` in a modern browser (or serve via any static server). Set `window.BACKEND_URL` in the console if backend runs on a different host/port.

### Next Steps
- Replace mock detectors with real models (MobileNetV2 head pose, YOLOv5)
- Add authentication and role-based access
- Persist violations and session logs to a database
- Build proctor dashboard for live alerts and analytics

# AI-Powered Real-Time Exam Monitoring System

## Overview
An AI-assisted online proctoring system that monitors webcam feeds for rule violations (head movement, multiple people, body visibility) and detects tab switching. Violations are logged and can be alerted to proctors via WhatsApp using Twilio.

## Architecture
- Backend: Flask (Python), SQLite (SQLAlchemy), OpenCV-based detectors, Twilio service
- Frontend: HTML/CSS/JS with React (via CDN), WebRTC webcam access
- Communication: REST API (JSON), base64-encoded JPEG frames every 2 seconds

## Features
- User login (simple username/password for demo)
- Head movement heuristics (left/right/down) using face bounding box geometry
- Multiple person detection using OpenCV HOG people detector
- Upper body visibility via OpenCV Haar upper-body cascade
- Tab switch detection via Page Visibility API and window focus/blur
- WhatsApp alert integration via Twilio (optional)
- SQLite logging of sessions and violations

## Prerequisites
- Python 3.10+
- Node.js is NOT required (frontend is static)
- Windows/macOS/Linux supported

## Quick Start

1) Create and activate a virtual environment
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
```

2) Install backend dependencies
```bash
pip install -r backend/requirements.txt
```

3) Configure environment (optional for Twilio)
Create a `.env` file in `backend/` or set environment variables in your shell:
```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TWILIO_WHATSAPP_TO=whatsapp:+91xxxxxxxxxx
SECRET_KEY=change-me
```
If Twilio vars are not set, alerts will be logged to console only.

4) Run backend API
```bash
python backend/app.py
```
The API runs at http://127.0.0.1:5000

5) Open frontend
Open `frontend/index.html` in a browser, or serve it via a simple HTTP server. For CORS camera permissions, it's best to use a local server:
```bash
python -m http.server 8080 -d frontend
```
Then visit http://127.0.0.1:8080

## Backend API
- `POST /api/auth/login` { username, password } -> { token }
- `POST /api/monitor/frame` { token, image_base64 } -> { violations: [...] }
- `POST /api/monitor/tab` { token, event, timestamp } -> { ok: true }
- `GET /api/violations` (Authorization: Bearer <token>) -> list
- `GET /api/health` -> { status: ok }

## Notes on AI Detectors
- Head movement detection uses face bounding box position/scale heuristics to approximate look left/right/down.
- Multiple people detection uses OpenCV HOG person detector; accuracy varies with camera placement.
- Upper body detection uses Haar cascade; may require good lighting.

You can later swap detectors for more advanced models (e.g., MediaPipe Face Mesh, Ultralytics YOLO) with minimal API changes.

## Development Scripts
- Re-run server with auto-reload during development:
```bash
pip install watchdog
python backend/app.py --reload
```

## Security
This demo uses a minimal token-based auth in-memory. For production, integrate a proper user system, HTTPS, CSRF protection, secure tokens (JWT), and stronger session management.

## License
MIT 