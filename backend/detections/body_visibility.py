from typing import Dict
import io
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None
try:
    from PIL import Image
except ImportError:
    Image = None


def _detect_upper_body_opencv(image_bytes: bytes) -> Dict:
    """Detect upper body visibility using OpenCV cascade classifiers"""
    if cv2 is None or np is None or Image is None:
        return {"upper_body_visible": True, "confidence": 0.8, "violation": False, "method": "fallback"}
    
    try:
        # Convert bytes to image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_img = np.array(img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Try to detect face first (indicates person is present)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        # Also try upper body detection (with fallback if file doesn't exist)
        try:
            upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
            upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
        except:
            upper_bodies = []  # Fallback if upper body cascade not available
        
        h, w = gray.shape
        
        # If we detect a face, assume upper body is visible (more lenient approach)
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = largest_face
            face_area = fw * fh
            total_area = h * w
            face_ratio = face_area / total_area
            
            # Check if face is positioned reasonably in the frame
            face_center_y = fy + fh // 2
            face_in_upper_half = face_center_y < h * 0.6  # Face should be in upper 60% of image
            
            # More lenient thresholds
            if face_ratio > 0.5:  # Only flag if face takes up more than 50% of image
                upper_body_visible = False
                confidence = 0.7
                violation = True
                reason = "face_too_large"
            elif not face_in_upper_half:
                upper_body_visible = False  
                confidence = 0.6
                violation = True
                reason = "face_positioned_low"
            else:
                # Face detected and reasonably positioned - assume body visible
                upper_body_visible = True
                confidence = 0.9
                violation = False
                reason = "face_detected_well_positioned"
                
        elif len(upper_bodies) > 0:
            # Upper body detected even without clear face
            upper_body_visible = True
            confidence = 0.8
            violation = False
            reason = "upper_body_detected"
            
        else:
            # No face or upper body detected - but be more lenient
            # Could be looking away temporarily or lighting issues
            upper_body_visible = True  # Changed to True to be more lenient
            confidence = 0.5
            violation = False  # Changed to False to reduce false positives
            reason = "no_detection_lenient"
        
        return {
            "upper_body_visible": upper_body_visible,
            "confidence": confidence,
            "violation": violation,
            "method": "opencv_cascade_improved",
            "faces_detected": len(faces),
            "upper_bodies_detected": len(upper_bodies),
            "reason": reason,
            "debug_info": {
                "image_size": f"{w}x{h}",
                "face_ratio": face_area / total_area if len(faces) > 0 else 0
            }
        }
    
    except Exception as e:
        return {
            "upper_body_visible": True,  # Default to no violation on error
            "confidence": 0.8,
            "violation": False,
            "method": "error_fallback",
            "error": str(e)
        }


def _simple_body_visibility_check(image_bytes: bytes) -> Dict:
    """Simple and lenient body visibility check"""
    if cv2 is None or np is None or Image is None:
        return {"upper_body_visible": True, "confidence": 0.9, "violation": False, "method": "simple_fallback"}
    
    try:
        # Convert bytes to image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_img = np.array(img)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        
        # Check for any face detection (very lenient)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20), maxSize=(w//2, h//2))
        
        # If we detect any face, assume body is visible
        if len(faces) > 0:
            return {
                "upper_body_visible": True,
                "confidence": 0.95,
                "violation": False,
                "method": "simple_face_detected",
                "faces_detected": len(faces)
            }
        
        # If no face detected, check for basic image content (not completely black/empty)
        # Calculate image brightness/activity
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # If there's reasonable image content, assume person is present but maybe looking away
        if mean_brightness > 10 and brightness_std > 5:  # Not a blank/black image
            return {
                "upper_body_visible": True,
                "confidence": 0.7,
                "violation": False,  # Don't penalize for looking away temporarily
                "method": "simple_image_content",
                "faces_detected": 0,
                "mean_brightness": float(mean_brightness)
            }
        
        # Only flag as violation if image is essentially empty/black
        return {
            "upper_body_visible": False,
            "confidence": 0.8,
            "violation": True,
            "method": "simple_no_content",
            "faces_detected": 0,
            "mean_brightness": float(mean_brightness)
        }
        
    except Exception as e:
        return {
            "upper_body_visible": True,  # Always default to no violation on error
            "confidence": 0.9,
            "violation": False,
            "method": "simple_error_fallback",
            "error": str(e)
        }


def infer_body_visibility(image_bytes: bytes) -> Dict:
    """Infer body visibility from image bytes - using simple, lenient approach"""
    return _simple_body_visibility_check(image_bytes)


