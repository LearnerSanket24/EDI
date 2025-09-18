import io
from typing import Dict, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


_face_mesh = None


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None and mp is not None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    return _face_mesh


def _bytes_to_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        if Image is None:
            return None
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return None
    if np is None or cv2 is None:
        return None
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _estimate_head_pose(bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    fm = _get_face_mesh()
    if fm is None:
        return None

    if cv2 is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None

    h, w = bgr.shape[:2]
    lms = res.multi_face_landmarks[0].landmark

    # Select 2D-3D correspondences (rough, standard indices)
    # Nose tip (1), Chin (152), Left eye corner (263), Right eye corner (33), Left mouth corner (287), Right mouth corner (57)
    idxs = [1, 152, 263, 33, 287, 57]
    pts_2d = np.array([[lms[i].x * w, lms[i].y * h] for i in idxs], dtype=np.float64)
    pts_3d = np.array([
        [0.0, 0.0, 0.0],      # nose
        [0.0, -63.6, -12.5],  # chin
        [43.3, 32.7, -26.0],  # left eye corner
        [-43.3, 32.7, -26.0], # right eye corner
        [28.9, -28.9, -24.1], # left mouth
        [-28.9, -28.9, -24.1] # right mouth
    ], dtype=np.float64)

    focal_length = w
    cam_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0

    pitch = np.degrees(x)
    yaw = np.degrees(y)
    return yaw, pitch


def _opencv_fallback_pose(bgr: np.ndarray) -> Dict:
    """Simple OpenCV-based head pose estimation using face detection"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Load cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {"pose": "unknown", "confidence": 0.0, "violation": False, "using_mediapipe": False}
        
        # Use the largest face
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        
        # Simple heuristic for head pose based on face position
        img_center_x = bgr.shape[1] // 2
        img_center_y = bgr.shape[0] // 2
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Determine head direction based on face position relative to center
        dx = face_center_x - img_center_x
        dy = face_center_y - img_center_y
        
        # Normalize by face size to get relative displacement
        dx_norm = dx / (w / 2) if w > 0 else 0
        dy_norm = dy / (h / 2) if h > 0 else 0
        
        pose = "forward"
        confidence = 0.5
        
        # Determine pose based on normalized displacement
        if abs(dx_norm) > abs(dy_norm):
            if dx_norm > 0.4:  # Face is to the right of center = looking left
                pose = "left" 
                confidence = min(0.9, 0.5 + abs(dx_norm) * 0.4)
            elif dx_norm < -0.4:  # Face is to the left of center = looking right
                pose = "right"
                confidence = min(0.9, 0.5 + abs(dx_norm) * 0.4)
        else:
            if dy_norm > 0.3:  # Face is lower = looking down
                pose = "down"
                confidence = min(0.9, 0.5 + abs(dy_norm) * 0.5)
            elif dy_norm < -0.3:  # Face is higher = looking up
                pose = "up"
                confidence = min(0.9, 0.5 + abs(dy_norm) * 0.5)
        
        violation = pose in ("left", "right", "down")
        
        return {
            "pose": pose,
            "head_pose": pose,
            "confidence": float(confidence),
            "violation": violation,
            "using_mediapipe": False,
            "method": "opencv_fallback",
            "face_location": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        }
        
    except Exception as e:
        return {
            "pose": "forward",
            "head_pose": "forward", 
            "confidence": 0.3, 
            "violation": False, 
            "using_mediapipe": False,
            "method": "opencv_fallback_error",
            "error": str(e)
        }

def infer_head_pose_mediapipe(image_bytes: bytes) -> Dict:
    bgr = _bytes_to_bgr(image_bytes)
    if bgr is None:
        return {"pose": "forward", "head_pose": "forward", "confidence": 0.3, "violation": False, "using_mediapipe": False}
    
    # Try MediaPipe first if available
    if mp is not None:
        try:
            est = _estimate_head_pose(bgr)
            if est is not None:
                yaw, pitch = est
                # Simple rule-based classification
                pose = "forward"
                if yaw <= -20:
                    pose = "left"
                elif yaw >= 20:
                    pose = "right"
                elif pitch >= 15:
                    pose = "down"
                elif pitch <= -15:
                    pose = "up"

                violation = pose in ("left", "right", "down")
                # Provide a heuristic confidence based on angle magnitude
                angle_mag = max(abs(yaw) / 40.0, abs(pitch) / 30.0)
                confidence = float(max(0.0, min(1.0, angle_mag)))
                return {
                    "pose": pose, 
                    "head_pose": pose,
                    "confidence": confidence, 
                    "violation": violation, 
                    "using_mediapipe": True,
                    "method": "mediapipe"
                }
        except Exception:
            pass  # Fall back to OpenCV
    
    # Fall back to OpenCV-based detection
    return _opencv_fallback_pose(bgr)



