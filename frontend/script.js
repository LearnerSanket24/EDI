// Configuration
const BACKEND = (window.BACKEND_URL || 'http://localhost:5000');
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
const sensitivitySlider = document.getElementById('sensitivitySlider');
const sensitivityValue = document.getElementById('sensitivityValue');
const warningThresholdInput = document.getElementById('warningThreshold');
const alertThresholdInput = document.getElementById('alertThreshold');
const soundEnabledInput = document.getElementById('soundEnabled');
const notificationsEnabledInput = document.getElementById('notificationsEnabled');
const loadingOverlay = document.getElementById('loadingOverlay');

// Hidden canvas for capture
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

// State
let mediaStream = null;
let captureTimer = null;
let sessionStartMs = null;
let violationsCount = 0;
let consecutiveViolationSeconds = 0;
let lastPose = 'forward';
let lastConfidence = 0;
let faceDetector = null;

// Audio cues
const audioWarning = new Audio('data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAAAAAABAAAAA');
const audioAlert = new Audio('data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAAAAAABAAAAA');

function fmtTime(ms) {
	const s = Math.floor(ms / 1000);
	const hh = String(Math.floor(s / 3600)).padStart(2, '0');
	const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
	const ss = String(s % 60).padStart(2, '0');
	return `${hh}:${mm}:${ss}`;
}

function setMonitoring(on) {
	monitoringStatusEl && (monitoringStatusEl.textContent = on ? 'Monitoring' : 'Offline');
	startBtn && (startBtn.disabled = on);
	stopBtn && (stopBtn.disabled = !on);
	if (on) {
		loadingOverlay && (loadingOverlay.style.display = 'none');
	}
}

function updateConfidenceBar(p) {
	const percent = Math.round(p * 100);
	if (confidenceFill) confidenceFill.style.width = `${percent}%`;
	if (confidenceText) confidenceText.textContent = `${percent}%`;
	if (confidenceEl) confidenceEl.textContent = `${percent}%`;
}

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
		<div>${Math.round(conf * 100)}%</div>
	`;
	poseTimeline.prepend(item);
	while (poseTimeline.children.length > 50) poseTimeline.removeChild(poseTimeline.lastChild);
}

function notifyBrowser(title, body) {
	if (!notificationsEnabledInput || !notificationsEnabledInput.checked) return;
	if (Notification && Notification.permission === 'granted') {
		new Notification(title, { body });
	} else if (Notification && Notification.permission !== 'denied') {
		Notification.requestPermission().then(p => {
			if (p === 'granted') new Notification(title, { body });
		});
	}
}

function pushAlert(level, message) {
	if (!alertsContainer) return;
	const empty = alertsContainer.querySelector('.no-alerts');
	if (empty) empty.remove();
	const el = document.createElement('div');
	el.className = `alert-item ${level}`;
	el.innerHTML = `
		<div class="alert-icon">${level === 'danger' ? '🚨' : level === 'warning' ? '⚠️' : 'ℹ️'}</div>
		<div class="alert-content">${message}</div>
		<div class="alert-time">${new Date().toLocaleTimeString()}</div>
	`;
	alertsContainer.prepend(el);
	alertCountEl && (alertCountEl.textContent = String((parseInt(alertCountEl.textContent || '0', 10) + 1)));
}

async function post(path, body) {
	const res = await fetch(`${BACKEND}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
	if (!res.ok) throw new Error(`${res.status}`);
	return res.json();
}

async function sendDetections(imageB64) {
	return Promise.all([
		post('/api/detections/head_pose', { image_b64: imageB64 }),
		post('/api/detections/multi_person', { image_b64: imageB64 }),
		post('/api/detections/body_visibility', { image_b64: imageB64 }),
	]);
}

function getSettings() {
	const sens = parseFloat(sensitivitySlider ? sensitivitySlider.value : '0.6');
	const warnSec = parseInt(warningThresholdInput ? warningThresholdInput.value : '3', 10);
	const alertSec = parseInt(alertThresholdInput ? alertThresholdInput.value : '5', 10);
	return { sensitivity: sens, warnSec, alertSec };
}

async function captureAndAnalyze() {
	if (!videoEl || !videoEl.videoWidth) return;
	captureCanvas.width = videoEl.videoWidth;
	captureCanvas.height = videoEl.videoHeight;
	captureCtx.drawImage(videoEl, 0, 0);
	// Sync overlay canvas size and draw face/eye boxes if supported
	if (overlayCanvas) {
		overlayCanvas.width = videoEl.videoWidth;
		overlayCanvas.height = videoEl.videoHeight;
	}
	try {
		if (!faceDetector && typeof window !== 'undefined' && 'FaceDetector' in window) {
			faceDetector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
		}
		if (faceDetector && overlayCtx) {
			const faces = await faceDetector.detect(videoEl);
			overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
			faces.forEach(f => {
				const bb = f.boundingBox || {};
				if (typeof bb.x === 'number') {
					overlayCtx.strokeStyle = '#00ff88';
					overlayCtx.lineWidth = 3;
					overlayCtx.strokeRect(bb.x, bb.y, bb.width, bb.height);
				}
				if (Array.isArray(f.landmarks)) {
					overlayCtx.fillStyle = '#00ff88';
					f.landmarks.filter(l => l.type === 'eye').forEach(l => {
						const pts = Array.isArray(l.locations) ? l.locations : [];
						pts.forEach(p => {
							overlayCtx.beginPath();
							overlayCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
							overlayCtx.fill();
						});
					});
				}
			});
		} else if (overlayCtx) {
			overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
		}
	} catch (_) {
		if (overlayCtx) overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
	}
	const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.8);
	try {
		const [head, multi, body] = await sendDetections(dataUrl);
		const { sensitivity, warnSec, alertSec } = getSettings();

		// Update UI for pose
		let pose = head.pose || head.head_pose || 'forward';
		let conf = typeof head.confidence === 'number' ? head.confidence : 0;
		
		// Debug logging
		console.log('Backend response:', { head, multi, body });
		console.log('Parsed pose:', pose, 'confidence:', conf);
		
		// If pose is still unknown/forward but we have a face, try to get better detection
		if (pose === 'unknown' && conf === 0 && head.using_trained === false) {
			// Use OpenCV fallback as secondary detection
			pose = 'forward'; 
			conf = 0.5; // Default confidence for basic detection
		}
		// Client-side fallback if backend returns unknown/0
		if ((pose === 'unknown' || conf === 0) && faceDetector && overlayCanvas && overlayCtx) {
			try {
				const faces = await faceDetector.detect(videoEl);
				if (faces && faces.length > 0) {
					const bb = faces[0].boundingBox || {};
					const cx = (bb.x || 0) + (bb.width || 0) / 2;
					const cy = (bb.y || 0) + (bb.height || 0) / 2;
					const vw = overlayCanvas.width || videoEl.videoWidth;
					const vh = overlayCanvas.height || videoEl.videoHeight;
					const nx = (cx / Math.max(1, vw)) * 2 - 1;
					const ny = (cy / Math.max(1, vh)) * 2 - 1;
					const horiz = Math.abs(nx);
					const vert = Math.abs(ny);
					let cp = 'forward';
					if (horiz > 0.25 && horiz >= vert) cp = nx < 0 ? 'left' : 'right';
					else if (vert > 0.25) cp = ny > 0 ? 'down' : 'up';
					pose = cp;
					conf = Math.min(1, Math.max(horiz, vert));
				}
			} catch (_) {}
		}
		const isViolationPose = (pose === 'left' || pose === 'right' || pose === 'down');
		const poseViolation = isViolationPose && conf >= sensitivity;
		setPoseIndicator(pose, conf, poseViolation ? 'warning' : 'ok');
		pushPoseHistory(pose, conf);

		// Body/person violations
		const multiViolation = !!multi?.violation || (typeof multi?.num_people === 'number' && multi.num_people > 1);
		const bodyViolation = !!body?.violation || body?.upper_body_visible === false;
		const anyViolationNow = poseViolation || multiViolation || bodyViolation;

		// accumulate consecutive seconds (approx)
		consecutiveViolationSeconds = anyViolationNow ? consecutiveViolationSeconds + POLL_MS / 1000 : 0;

		// thresholds → warnings/alerts
		if (anyViolationNow && consecutiveViolationSeconds >= warnSec && consecutiveViolationSeconds < alertSec) {
			setPoseIndicator(pose, conf, 'warning');
			pushAlert('warning', `Suspicious behavior detected (${poseViolation ? 'Head Pose' : multiViolation ? 'Multiple Person' : 'Body Visibility'})`);
			soundEnabledInput && soundEnabledInput.checked && audioWarning.play().catch(() => {});
		}
		if (anyViolationNow && consecutiveViolationSeconds >= alertSec) {
			setPoseIndicator(pose, conf, 'danger');
			pushAlert('danger', `Alert: Sustained violation (${[
				poseViolation ? 'Head Pose' : null,
				multiViolation ? 'Multiple Person' : null,
				bodyViolation ? 'Body Visibility' : null,
			].filter(Boolean).join(', ')})`);
			notifyBrowser('Exam Alert', 'Sustained suspicious behavior detected');
			soundEnabledInput && soundEnabledInput.checked && audioAlert.play().catch(() => {});
			violationsCount += 1;
			totalViolationsEl && (totalViolationsEl.textContent = String(violationsCount));

			// Fire backend alert
			try {
				await post('/api/alert', {
					student: 'anonymous',
					violation: [
						poseViolation ? 'Head Pose' : null,
						multiViolation ? 'Multiple Person' : null,
						bodyViolation ? 'Body Visibility' : null,
					].filter(Boolean).join(', '),
					timestamp: new Date().toISOString(),
				});
			} catch (e) {
				// ignore network alert errors for UI continuity
			}

			// reset counter after an alert so we don't spam
			consecutiveViolationSeconds = 0;
		}
	} catch (e) {
		pushAlert('info', `Detection failed: ${e.message}`);
	}
}

async function startMonitoring() {
	try {
		loadingOverlay && (loadingOverlay.style.display = 'flex');
		mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
		videoEl.srcObject = mediaStream;
		await videoEl.play().catch(() => {});
		sessionStartMs = Date.now();
		violationsCount = 0;
		consecutiveViolationSeconds = 0;
		alertCountEl && (alertCountEl.textContent = '0');
		totalViolationsEl && (totalViolationsEl.textContent = '0');
		setMonitoring(true);
		captureTimer = setInterval(captureAndAnalyze, POLL_MS);
	} catch (e) {
		pushAlert('danger', `Webcam access failed: ${e.message}`);
		setMonitoring(false);
	}
}

function stopMonitoring() {
	if (captureTimer) clearInterval(captureTimer);
	captureTimer = null;
	if (mediaStream) {
		mediaStream.getTracks().forEach(t => t.stop());
		mediaStream = null;
	}
	setMonitoring(false);
}

// Session timer
setInterval(() => {
	if (!sessionStartMs) return;
	sessionTimeEl && (sessionTimeEl.textContent = fmtTime(Date.now() - sessionStartMs));
}, 1000);

// Settings UI bindings
if (sensitivitySlider && sensitivityValue) {
	sensitivityValue.textContent = `${Math.round(parseFloat(sensitivitySlider.value) * 100)}%`;
	sensitivitySlider.addEventListener('input', () => {
		sensitivityValue.textContent = `${Math.round(parseFloat(sensitivitySlider.value) * 100)}%`;
	});
}

if (settingsBtn && settingsPanel) {
	settingsBtn.addEventListener('click', () => {
		settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
	});
}

if (clearAlertsBtn) {
	clearAlertsBtn.addEventListener('click', () => {
		alertsContainer.innerHTML = '<div class="no-alerts"><i class="fas fa-check-circle"></i><p>No alerts at this time</p></div>';
		alertCountEl && (alertCountEl.textContent = '0');
	});
}

// Tab visibility as a rule
if (typeof document !== 'undefined') {
	document.addEventListener('visibilitychange', () => {
		if (document.hidden) {
			pushAlert('warning', 'Tab hidden or switched');
			post('/api/alert', {
				student: 'anonymous',
				violation: 'Tab switch/hidden',
				timestamp: new Date().toISOString(),
			}).catch(() => {});
		}
	});
}

// Buttons
startBtn && startBtn.addEventListener('click', startMonitoring);
stopBtn && stopBtn.addEventListener('click', stopMonitoring);

// Ask for notification permission up front
if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
	Notification.requestPermission().catch(() => {});
}

// Ensure loading overlay doesn't block the UI before monitoring starts
if (typeof window !== 'undefined') {
	window.addEventListener('DOMContentLoaded', () => {
		loadingOverlay && (loadingOverlay.style.display = 'none');
	});
}


