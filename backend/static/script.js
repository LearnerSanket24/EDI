// Configuration
const BACKEND = window.location.origin; // Use same origin as the page
const POLL_MS = 1500; // capture interval

// DOM refs
const videoEl = document.getElementById('videoElement') || document.getElementById('video');
const overlayCanvas = document.getElementById('overlayCanvas');
const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null;
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const monitoringStatusEl = document.getElementById('monitoringStatus');
const alertCountEl = document.getElementById('alertCount');
const poseIndicator = document.getElementById('poseIndicator');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const totalViolationsEl = document.getElementById('totalViolations');
const currentPoseEl = document.getElementById('currentPose');
const sessionTimeEl = document.getElementById('sessionTime');
const confidenceEl = document.getElementById('confidence');
const poseTimeline = document.getElementById('poseTimeline');
const alertsContainer = document.getElementById('alertsContainer');
const clearAlertsBtn = document.getElementById('clearAlertsBtn');
const settingsBtn = document.getElementById('settingsBtn');
const settingsPanel = document.getElementById('settingsPanel');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const sensitivitySlider = document.getElementById('sensitivitySlider');
const sensitivityValue = document.getElementById('sensitivityValue');
const warningThresholdInput = document.getElementById('warningThreshold');
const alertThresholdInput = document.getElementById('alertThreshold');
const soundEnabledInput = document.getElementById('soundEnabled');
const notificationsEnabledInput = document.getElementById('notificationsEnabled');
const loadingOverlay = document.getElementById('loadingOverlay');
const backendStatusEl = document.getElementById('backendStatus');
const headPoseModelStatusEl = document.getElementById('headPoseModelStatus');
const personDetectionStatusEl = document.getElementById('personDetectionStatus');
const cameraStatusEl = document.getElementById('cameraStatus');

// Hidden canvas for capture
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

// State
let mediaStream = null;
let captureTimer = null;
let inflight = false;
let sessionStartMs = null;
let violationsCount = 0;
let consecutiveViolationSeconds = 0;
let lastAlertAtMs = 0;
const ALERT_COOLDOWN_MS = 4000;
let lastPose = 'forward';
let lastConfidence = 0;
let alertsCount = 0;

// Settings
let settings = {
    sensitivity: 5,
    warningThreshold: 3,
    alertThreshold: 10,
    soundEnabled: true,
    notificationsEnabled: true
};

// Audio cues
const audioWarning = new Audio('data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAAAAAABAAAAA');
const audioAlert = new Audio('data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAAAAAABAAAAA');

// Format time for display
function fmtTime(ms) {
    const s = Math.floor(ms / 1000);
    const hh = String(Math.floor(s / 3600)).padStart(2, '0');
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
}

// Update monitoring status UI
function setMonitoring(on) {
    monitoringStatusEl && (monitoringStatusEl.textContent = on ? 'Monitoring' : 'Offline');
    startBtn && (startBtn.disabled = on);
    stopBtn && (stopBtn.disabled = !on);
    if (on) {
        loadingOverlay && (loadingOverlay.style.display = 'none');
    }
}

// Update confidence bar UI
function updateConfidenceBar(p) {
    const percent = Math.round(p * 100);
    if (confidenceFill) confidenceFill.style.width = `${percent}%`;
    if (confidenceText) confidenceText.textContent = `${percent}%`;
    if (confidenceEl) confidenceEl.textContent = `${percent}%`;
}

// Update pose indicator UI
function setPoseIndicator(pose, conf, level = 'ok') {
    lastPose = pose;
    lastConfidence = conf;
    if (!poseIndicator) return;
    poseIndicator.classList.remove('warning', 'danger');
    if (level === 'warning') poseIndicator.classList.add('warning');
    if (level === 'danger') poseIndicator.classList.add('danger');
    const pretty = pose.charAt(0).toUpperCase() + pose.slice(1);
    poseIndicator.querySelector('span').textContent =
        (level === 'ok' ? 'Looking ' : level === 'warning' ? 'Warning: ' : 'Alert: ') + (pose === 'forward' ? 'Forward' : pretty);
    updateConfidenceBar(conf);
    if (currentPoseEl) currentPoseEl.textContent = pretty;
}

// Add pose to history timeline
function pushPoseHistory(pose, conf) {
    if (!poseTimeline) return;
    const item = document.createElement('div');
    item.className = 'pose-item fade-in';
    const time = new Date().toLocaleTimeString();
    const pretty = pose.charAt(0).toUpperCase() + pose.slice(1);
    const tagClass = `pose-type ${pose}`;
    item.innerHTML = `
        <div>
            <div class="pose-time">${time}</div>
            <div class="pose-type ${pose}">${pretty}</div>
        </div>
        <div class="pose-confidence">${Math.round(conf * 100)}% confidence</div>
    `;
    poseTimeline.prepend(item);
    
    // Limit history items
    if (poseTimeline.children.length > 20) {
        poseTimeline.removeChild(poseTimeline.lastChild);
    }
}

// Add alert to alerts container
function pushAlert(message, type = 'violation') {
    if (!alertsContainer) return;
    alertsCount++;
    if (alertCountEl) alertCountEl.textContent = alertsCount;
    
    const item = document.createElement('div');
    item.className = 'alert-item fade-in';
    const time = new Date().toLocaleTimeString();
    item.innerHTML = `
        <div>
            <div class="alert-time">${time}</div>
            <div class="alert-message">${message}</div>
        </div>
        <div class="alert-actions">
            <button class="btn btn-small btn-secondary dismiss-alert">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add dismiss functionality
    const dismissBtn = item.querySelector('.dismiss-alert');
    dismissBtn.addEventListener('click', () => {
        item.remove();
        alertsCount--;
        if (alertCountEl) alertCountEl.textContent = alertsCount;
    });
    
    alertsContainer.prepend(item);
    
    // Play sound if enabled
    if (settings.soundEnabled) {
        if (type === 'warning') {
            audioWarning.play().catch(() => {});
        } else {
            audioAlert.play().catch(() => {});
        }
    }
    
    // Show browser notification if enabled
    if (settings.notificationsEnabled && "Notification" in window) {
        if (Notification.permission === "granted") {
            new Notification("AI Exam Monitor", {
                body: message,
                icon: "/favicon.ico"
            });
        } else if (Notification.permission !== "denied") {
            Notification.requestPermission().then(permission => {
                if (permission === "granted") {
                    new Notification("AI Exam Monitor", {
                        body: message,
                        icon: "/favicon.ico"
                    });
                }
            });
        }
    }
}

// Clear all alerts
function clearAlerts() {
    if (!alertsContainer) return;
    alertsContainer.innerHTML = '';
    alertsCount = 0;
    if (alertCountEl) alertCountEl.textContent = alertsCount;
}

// Update session time
function updateSessionTime() {
    if (!sessionTimeEl || !sessionStartMs) return;
    const elapsed = Date.now() - sessionStartMs;
    sessionTimeEl.textContent = fmtTime(elapsed);
}

// Check backend system status
async function checkSystemStatus() {
    const controller = new AbortController();
    const to = setTimeout(() => controller.abort(), 5000);
    try {
        const response = await fetch(`${BACKEND}/api/system/status`, { signal: controller.signal });
        if (!response.ok) throw new Error('Backend not responding');
        
        const data = await response.json();
        clearTimeout(to);
        
        // Update backend status
        if (backendStatusEl) {
            backendStatusEl.textContent = 'Connected';
            backendStatusEl.className = 'status-value online';
        }
        
        // Update head pose model status
        if (headPoseModelStatusEl) {
            const headPoseAvailable = data.capabilities?.head_pose_detection || false;
            headPoseModelStatusEl.textContent = headPoseAvailable ? 'Loaded' : 'Not Available';
            headPoseModelStatusEl.className = `status-value ${headPoseAvailable ? 'online' : 'offline'}`;
        }
        
        // Update person detection status
        if (personDetectionStatusEl) {
            const personDetectionAvailable = data.capabilities?.multi_person_detection || false;
            personDetectionStatusEl.textContent = personDetectionAvailable ? 'Loaded' : 'Not Available';
            personDetectionStatusEl.className = `status-value ${personDetectionAvailable ? 'online' : 'offline'}`;
        }
        
        return true;
    } catch (error) {
        console.error('Error checking system status:', error);
        
        // Update backend status
        if (backendStatusEl) {
            backendStatusEl.textContent = 'Disconnected';
            backendStatusEl.className = 'status-value offline';
        }
        clearTimeout(to);
        return false;
    }
}

// Capture frame and send to backend
async function captureAndProcess() {
    if (!videoEl || !mediaStream) return;
    if (inflight) return;
    inflight = true;
    
    // Downscale to reduce bandwidth/CPU
    const targetW = 640;
    const scale = Math.min(1, targetW / Math.max(1, videoEl.videoWidth));
    const w = Math.floor(videoEl.videoWidth * scale);
    const h = Math.floor(videoEl.videoHeight * scale);
    captureCanvas.width = w;
    captureCanvas.height = h;
    
    // Draw video frame to canvas
    captureCtx.drawImage(videoEl, 0, 0, w, h);
    
    // Convert to base64
    const imageBase64 = captureCanvas.toDataURL('image/jpeg', 0.7);
    
    try {
        // Send to unified detection endpoint with timeout
        const controller = new AbortController();
        const to = setTimeout(() => controller.abort(), 10000);
        const response = await fetch(`${BACKEND}/api/detections/unified`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_b64: imageBase64, user_id: 'web-client' }),
            signal: controller.signal
        });
        clearTimeout(to);
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const result = await response.json();
        
        // Process head pose results
        if (result.head_pose) {
            const { direction, confidence, violation } = result.head_pose;
            
            // Update UI
            let level = 'ok';
            if (violation) {
                consecutiveViolationSeconds += POLL_MS / 1000;
                const now = Date.now();
                if (consecutiveViolationSeconds >= settings.alertThreshold) {
                    level = 'danger';
                    if (now - lastAlertAtMs > ALERT_COOLDOWN_MS) {
                        pushAlert(`Looking ${direction} for ${Math.round(consecutiveViolationSeconds)} seconds`, 'violation');
                        lastAlertAtMs = now;
                        violationsCount++;
                        if (totalViolationsEl) totalViolationsEl.textContent = violationsCount;
                    }
                } else if (consecutiveViolationSeconds >= settings.warningThreshold) {
                    level = 'warning';
                }
            } else {
                consecutiveViolationSeconds = 0;
            }
            
            setPoseIndicator(direction, confidence, level);
            pushPoseHistory(direction, confidence);
        }
        
        // Process multi-person results
        if (result.multi_person) {
            const mp = result.multi_person;
            if (mp.violation) {
                const now = Date.now();
                if (now - lastAlertAtMs > ALERT_COOLDOWN_MS) {
                    pushAlert(`Multiple people detected: ${mp.num_people}`, 'violation');
                    lastAlertAtMs = now;
                    violationsCount++;
                    if (totalViolationsEl) totalViolationsEl.textContent = violationsCount;
                }
            }
        }
        
        // Process body visibility results
        if (result.body_visibility && result.body_visibility.violation) {
            const now = Date.now();
            if (now - lastAlertAtMs > ALERT_COOLDOWN_MS) {
                pushAlert('Body not visible in frame', 'violation');
                lastAlertAtMs = now;
                violationsCount++;
                if (totalViolationsEl) totalViolationsEl.textContent = violationsCount;
            }
        }
        
    } catch (error) {
        console.error('Error processing frame:', error);
    } finally {
        inflight = false;
    }
}

// Start monitoring
async function startMonitoring() {
    try {
        // Check if backend is available
        const systemAvailable = await checkSystemStatus();
        if (!systemAvailable) {
            alert('Backend system is not available. Please check your connection.');
            return;
        }
        
        // Request camera access
        mediaStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            } 
        });
        
        // Update camera status
        if (cameraStatusEl) {
            cameraStatusEl.textContent = 'Connected';
            cameraStatusEl.className = 'status-value online';
        }
        
        // Connect video element to stream
        if (videoEl) {
            videoEl.srcObject = mediaStream;
            await videoEl.play();
        }
        
        // Start session timer
        sessionStartMs = Date.now();
        setInterval(updateSessionTime, 1000);
        
        // Start capture timer
        captureTimer = setInterval(captureAndProcess, POLL_MS);

        // Kick backend warmup (non-blocking)
        try { fetch(`${BACKEND}/api/warmup`, { method: 'POST' }); } catch (_) {}
        
        // Update UI
        setMonitoring(true);
        
    } catch (error) {
        console.error('Error starting monitoring:', error);
        alert('Could not access camera. Please check permissions and try again.');
        
        // Update camera status
        if (cameraStatusEl) {
            cameraStatusEl.textContent = 'Access Denied';
            cameraStatusEl.className = 'status-value offline';
        }
    }
}

// Stop monitoring
function stopMonitoring() {
    // Stop capture timer
    if (captureTimer) {
        clearInterval(captureTimer);
        captureTimer = null;
    }
    
    // Stop media stream
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    // Clear video element
    if (videoEl) {
        videoEl.srcObject = null;
    }
    
    // Update camera status
    if (cameraStatusEl) {
        cameraStatusEl.textContent = 'Disconnected';
        cameraStatusEl.className = 'status-value offline';
    }
    
    // Update UI
    setMonitoring(false);
}

// Toggle settings panel
function toggleSettings() {
    if (settingsPanel) {
        settingsPanel.classList.toggle('active');
    }
}

// Save settings
function saveSettings() {
    if (sensitivitySlider) settings.sensitivity = parseInt(sensitivitySlider.value);
    if (warningThresholdInput) settings.warningThreshold = parseInt(warningThresholdInput.value);
    if (alertThresholdInput) settings.alertThreshold = parseInt(alertThresholdInput.value);
    if (soundEnabledInput) settings.soundEnabled = soundEnabledInput.checked;
    if (notificationsEnabledInput) settings.notificationsEnabled = notificationsEnabledInput.checked;
    
    // Save to localStorage
    localStorage.setItem('examMonitorSettings', JSON.stringify(settings));
}

// Load settings
function loadSettings() {
    const savedSettings = localStorage.getItem('examMonitorSettings');
    if (savedSettings) {
        settings = { ...settings, ...JSON.parse(savedSettings) };
    }
    
    // Update UI
    if (sensitivitySlider) {
        sensitivitySlider.value = settings.sensitivity;
        sensitivityValue.textContent = settings.sensitivity;
    }
    if (warningThresholdInput) warningThresholdInput.value = settings.warningThreshold;
    if (alertThresholdInput) alertThresholdInput.value = settings.alertThreshold;
    if (soundEnabledInput) soundEnabledInput.checked = settings.soundEnabled;
    if (notificationsEnabledInput) notificationsEnabledInput.checked = settings.notificationsEnabled;
}

// Initialize
async function init() {
    // Load settings
    loadSettings();
    
    // Check system status (non-blocking)
    checkSystemStatus();
    
    // Add event listeners
    if (startBtn) startBtn.addEventListener('click', startMonitoring);
    if (stopBtn) stopBtn.addEventListener('click', stopMonitoring);
    if (settingsBtn) settingsBtn.addEventListener('click', toggleSettings);
    if (closeSettingsBtn) closeSettingsBtn.addEventListener('click', toggleSettings);
    if (clearAlertsBtn) clearAlertsBtn.addEventListener('click', clearAlerts);
    
    // Settings change listeners
    if (sensitivitySlider) {
        sensitivitySlider.addEventListener('input', () => {
            sensitivityValue.textContent = sensitivitySlider.value;
        });
        sensitivitySlider.addEventListener('change', saveSettings);
    }
    if (warningThresholdInput) warningThresholdInput.addEventListener('change', saveSettings);
    if (alertThresholdInput) alertThresholdInput.addEventListener('change', saveSettings);
    if (soundEnabledInput) soundEnabledInput.addEventListener('change', saveSettings);
    if (notificationsEnabledInput) notificationsEnabledInput.addEventListener('change', saveSettings);
    
    // Hide loading overlay
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

// Start initialization when DOM is loaded
document.addEventListener('DOMContentLoaded', init);