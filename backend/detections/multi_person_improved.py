from typing import Dict, List, Optional
import base64
import io
import os
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ImprovedMultiPersonDetector:
    """
    Improved multi-person detector with better accuracy and debugging
    """
    
    def __init__(self):
        self._custom_model = None
        self._yolo_model = None
        self._model_loaded = False
        self._debug_mode = True
    
    def _load_models(self):
        """Load both custom and fallback models"""
        if YOLO is None:
            print("⚠️ YOLO not available")
            return False
        
        # Try to load custom model
        try:
            self._custom_model = YOLO(os.path.join(os.path.dirname(__file__), '..', 'models', 'crowdhuman_custom.pt'))
            print("✓ Loaded custom CrowdHuman model")
        except Exception as e:
            print(f"⚠️ Could not load custom model: {e}")
            self._custom_model = None
        
        # Load YOLOv8n as fallback
        try:
            self._yolo_model = YOLO('yolov8n.pt')
            print("✓ Loaded YOLOv8n fallback model")
            self._model_loaded = True
        except Exception as e:
            print(f"✗ Could not load fallback model: {e}")
            self._yolo_model = None
        
        return self._model_loaded
    
    def _get_best_model(self):
        """Get the best available model"""
        if not self._model_loaded:
            self._load_models()
        
        # Prefer your custom trained CrowdHuman model
        if self._custom_model is not None:
            return self._custom_model, "CrowdHuman Custom"
        elif self._yolo_model is not None:
            return self._yolo_model, "YOLOv8n Fallback"
        else:
            return None, "None"
    
    def _bytes_to_numpy(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to numpy array"""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return np.array(img)
        except Exception as e:
            if self._debug_mode:
                print(f"Error converting image: {e}")
            return None
    
    def _filter_person_detections(self, boxes, confidences, img_shape) -> List[Dict]:
        """Filter and validate person detections"""
        if boxes is None or len(boxes) == 0:
            return []
        
        height, width = img_shape[:2]
        valid_detections = []
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            det_width = x2 - x1
            det_height = y2 - y1
            
            # Basic size filtering
            if det_width < 12 or det_height < 24:  # Relaxed small filter
                continue
            
            if det_width > width * 0.95 or det_height > height * 0.95:  # Too large
                continue
            
            # Aspect ratio filtering for persons (should be taller than wide)
            aspect_ratio = det_height / det_width if det_width > 0 else 0
            if aspect_ratio < 0.8 or aspect_ratio > 6.0:  # Relaxed aspect range
                continue
            
            # Position filtering (person shouldn't be at very edge)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Skip detections too close to edges (might be partial)
            margin = 0.02  # Relaxed to 2%
            if (center_x < width * margin or center_x > width * (1 - margin) or
                center_y < height * margin or center_y > height * (1 - margin)):
                continue
            
            valid_detections.append({
                "person_id": len(valid_detections) + 1,
                "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                "center": {"x": float(center_x), "y": float(center_y)},
                "size": {"width": float(det_width), "height": float(det_height)},
                "confidence": float(conf),
                "aspect_ratio": float(aspect_ratio)
            })
        
        return valid_detections
    
    def detect_persons(self, image_bytes: bytes) -> Dict:
        """Detect persons in image with improved accuracy"""
        model, model_type = self._get_best_model()
        
        if model is None:
            return {
                "num_people": 0,
                "confidence": 0.0,
                "violation": False,
                "model_type": "None",
                "error": "No models available",
                "people_locations": []
            }
        
        frame = self._bytes_to_numpy(image_bytes)
        if frame is None:
            return {
                "num_people": 0,
                "confidence": 0.0,
                "violation": False,
                "model_type": model_type,
                "error": "Could not process image",
                "people_locations": []
            }
        
        try:
            # Use appropriate confidence threshold for each model
            if "Custom" in model_type:
                conf_threshold = 0.25
            else:
                conf_threshold = 0.25
            
            if self._debug_mode:
                print(f"Using {model_type} model with confidence threshold {conf_threshold}")
            
            # Run inference
            results = model.predict(
                source=frame,
                classes=[0],  # Only person class
                conf=conf_threshold,
                iou=0.4,  # NMS IoU threshold
                verbose=False,
                save=False
            )
            
            if not results or len(results) == 0:
                return {
                    "num_people": 0,
                    "confidence": 0.0,
                    "violation": False,
                    "model_type": model_type,
                    "people_locations": []
                }
            
            r = results[0]
            raw_detections = len(r.boxes) if r.boxes is not None else 0
            
            if self._debug_mode:
                print(f"Raw detections from {model_type}: {raw_detections}")
            
            # Filter detections
            people_locations = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                
                people_locations = self._filter_person_detections(
                    boxes, confidences, frame.shape
                )
            
            num_people = len(people_locations)
            max_conf = max([p["confidence"] for p in people_locations]) if people_locations else 0.0
            violation = num_people > 1
            
            if self._debug_mode:
                print(f"Final detections after filtering: {num_people}")
                if num_people > 0:
                    print(f"Max confidence: {max_conf:.3f}")
                    print(f"Violation detected: {violation}")
            
            return {
                "num_people": num_people,
                "confidence": max_conf,
                "violation": violation,
                "model_type": model_type,
                "people_locations": people_locations,
                "debug_info": {
                    "confidence_threshold": conf_threshold,
                    "raw_detections": raw_detections,
                    "filtered_detections": num_people,
                    "image_shape": frame.shape
                }
            }
            
        except Exception as e:
            error_msg = str(e)
            if self._debug_mode:
                print(f"Error in {model_type} detection: {error_msg}")
            
            return {
                "num_people": 0,
                "confidence": 0.0,
                "violation": False,
                "model_type": model_type,
                "error": error_msg,
                "people_locations": []
            }


# Global detector instance
_detector = None

def get_detector() -> ImprovedMultiPersonDetector:
    """Get or create detector instance"""
    global _detector
    if _detector is None:
        _detector = ImprovedMultiPersonDetector()
    return _detector

def infer_multi_person_improved(image_bytes: bytes) -> Dict:
    """Improved multi-person detection function"""
    detector = get_detector()
    return detector.detect_persons(image_bytes)
