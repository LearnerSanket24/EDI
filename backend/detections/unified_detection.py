from typing import Dict, List, Tuple, Optional
import base64
import io
import os
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Import existing modules
from .head_pose import infer_head_pose
from .body_visibility import infer_body_visibility

class UnifiedDetectionSystem:
    """
    Unified detection system that combines:
    1. Custom CrowdHuman multi-person detection
    2. Head pose detection 
    3. Body visibility detection
    """
    
    def __init__(self):
        self._custom_model = None
        self._fallback_model = None
        self._model_loaded = False
        self._last_error = None
    
    def _get_custom_model(self):
        """Load custom CrowdHuman model with fallback"""
        if self._custom_model is None and YOLO is not None:
            try:
                # Try to load custom trained model from multiple possible locations
                model_paths = [
                    'models/crowdhuman_custom.pt',
                    os.path.join(os.path.dirname(__file__), '..', 'models', 'crowdhuman_custom.pt'),
                    os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'crowdhuman_custom.pt'),
                    os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'training_results_light', 'crowdhuman_light_20250911_114753', 'weights', 'best.pt')
                ]
                
                for path in model_paths:
                    if os.path.exists(path):
                        self._custom_model = YOLO(path)
                        print(f"✓ Loaded custom CrowdHuman model from {path}")
                        self._model_loaded = True
                        return self._custom_model
                
                print("⚠ Could not find custom model in any expected location")
                self._custom_model = False
                self._last_error = "Model file not found in any expected location"
            except Exception as e:
                print(f"⚠ Could not load custom model: {e}")
                self._custom_model = False
                self._last_error = str(e)
        
        # Try fallback model
        if self._custom_model is False or self._custom_model is None:
            if self._fallback_model is None and YOLO is not None:
                try:
                    # Try multiple possible locations for fallback model
                    fallback_paths = [
                        'yolov8n.pt',
                        os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt'),
                        os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt')
                    ]
                    
                    for path in fallback_paths:
                        if os.path.exists(path):
                            self._fallback_model = YOLO(path)
                            print(f"✓ Loaded fallback YOLOv8n model from {path}")
                            self._model_loaded = True
                            return self._fallback_model
                    
                    # If no local file found, try to download from ultralytics
                    self._fallback_model = YOLO('yolov8n.pt')
                    print("✓ Downloaded fallback YOLOv8n model")
                    self._model_loaded = True
                    return self._fallback_model
                except Exception as e:
                    print(f"✗ Could not load fallback model: {e}")
                    self._last_error = str(e)
                    return None
            return self._fallback_model
        
        return self._custom_model
    
    def _bytes_to_numpy(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to numpy array"""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return np.array(img)
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1["bbox"]["x1"], box1["bbox"]["y1"], box1["bbox"]["x2"], box1["bbox"]["y2"]
        x1_2, y1_2, x2_2, y2_2 = box2["bbox"]["x1"], box2["bbox"]["y1"], box2["bbox"]["x2"], box2["bbox"]["y2"]
        
        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _apply_custom_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Sort detections by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while sorted_detections:
            # Take the detection with highest confidence
            current = sorted_detections.pop(0)
            keep.append(current)
            
            # Remove detections that have high IoU with current detection
            remaining = []
            for detection in sorted_detections:
                iou = self._calculate_iou(current, detection)
                if iou < iou_threshold:
                    remaining.append(detection)
            
            sorted_detections = remaining
        
        # Reassign person IDs
        for i, detection in enumerate(keep):
            detection["person_id"] = i + 1
        
        return keep
    
    def detect_multiple_persons(self, image_bytes: bytes) -> Dict:
        """Enhanced multi-person detection using custom CrowdHuman model"""
        model = self._get_custom_model()
        using_custom = (model == self._custom_model and model is not None)
        
        if model is None:
            return {
                "num_people": 0, 
                "confidence": 0.0, 
                "violation": False, 
                "using_custom": False,
                "error": f"No models available: {self._last_error}",
                "people_locations": []
            }

        frame = self._bytes_to_numpy(image_bytes)
        if frame is None:
            return {
                "num_people": 0, 
                "confidence": 0.0, 
                "violation": False, 
                "using_custom": using_custom,
                "error": "Could not process image",
                "people_locations": []
            }
        
        try:
            # Use more conservative confidence thresholds to reduce false positives
            if using_custom:
                conf_threshold = 0.25
            else:
                conf_threshold = 0.25
            
            results = model.predict(source=frame, classes=[0], conf=conf_threshold, verbose=False)
            
            if not results:
                return {
                    "num_people": 0, 
                    "confidence": 0.0, 
                    "violation": False, 
                    "using_custom": using_custom,
                    "people_locations": []
                }

            r = results[0]
            
            people_locations = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = r.boxes.conf.cpu().numpy()
                
                # Apply additional filtering to reduce false positives
                valid_detections = []
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Filter out detections that are too small or have wrong aspect ratio
                    # Typical person aspect ratio should be height > width
                    min_width = 10  # relaxed minimum width in pixels
                    min_height = 20  # relaxed minimum height in pixels
                    max_aspect_ratio = 6.0  # relaxed maximum ratio
                    min_aspect_ratio = 0.8  # relaxed minimum ratio
                    
                    if width < min_width or height < min_height:
                        continue  # Skip too small detections
                    
                    aspect_ratio = height / width if width > 0 else 0
                    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                        continue  # Skip detections with wrong aspect ratio
                    
                    # Skip detections with very low confidence even after initial filtering
                    if conf < conf_threshold:
                        continue
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detection = {
                        "person_id": len(valid_detections) + 1,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "center": {"x": float(center_x), "y": float(center_y)},
                        "size": {"width": float(width), "height": float(height)},
                        "confidence": float(conf),
                        "aspect_ratio": float(aspect_ratio)
                    }
                    valid_detections.append(detection)
                
                # Apply Non-Maximum Suppression manually if needed to remove overlapping detections
                people_locations = self._apply_custom_nms(valid_detections, iou_threshold=0.5)
            
            num_people = len(people_locations)
            max_conf = max([p["confidence"] for p in people_locations]) if people_locations else 0.0
            violation = num_people > 1
            
            # Add debug information
            debug_info = {
                "confidence_threshold": conf_threshold,
                "raw_detections": len(r.boxes) if r.boxes is not None else 0,
                "filtered_detections": num_people,
                "model_used": "CrowdHuman Custom" if using_custom else "YOLOv8n Fallback"
            }
            
            return {
                "num_people": num_people,
                "confidence": max_conf,
                "violation": violation,
                "using_custom": using_custom,
                "model_type": "CrowdHuman Custom" if using_custom else "YOLOv8n Fallback",
                "people_locations": people_locations,
                "debug_info": debug_info
            }
            
        except Exception as e:
            return {
                "num_people": 0, 
                "confidence": 0.0, 
                "violation": False, 
                "using_custom": using_custom,
                "error": str(e),
                "people_locations": []
            }
    
    def unified_inference(self, image_bytes: bytes) -> Dict:
        """
        Run all detection models on the image and return combined results
        """
        results = {
            "timestamp": "",
            "image_processed": True,
            "detections": {},
            "violations": {},
            "summary": {}
        }
        
        try:
            import datetime
            results["timestamp"] = datetime.datetime.now().isoformat()
            
            # 1. Multi-person detection
            print("Running multi-person detection...")
            multi_person_result = self.detect_multiple_persons(image_bytes)
            results["detections"]["multi_person"] = multi_person_result
            
            # 2. Head pose detection
            print("Running head pose detection...")
            head_pose_result = infer_head_pose(image_bytes)
            # Normalize head pose keys so frontend always receives 'direction'
            normalized_head_pose = dict(head_pose_result)
            if "direction" not in normalized_head_pose:
                if "head_pose" in normalized_head_pose:
                    normalized_head_pose["direction"] = normalized_head_pose.get("head_pose")
                elif "pose" in normalized_head_pose:
                    normalized_head_pose["direction"] = normalized_head_pose.get("pose")
            results["detections"]["head_pose"] = normalized_head_pose
            
            # 3. Body visibility detection
            print("Running body visibility detection...")
            body_visibility_result = infer_body_visibility(image_bytes)
            results["detections"]["body_visibility"] = body_visibility_result

            # Also expose top-level keys expected by new frontend
            results["multi_person"] = multi_person_result
            results["head_pose"] = normalized_head_pose
            results["body_visibility"] = body_visibility_result
            
            # Analyze violations
            violations = []
            violation_details = {}
            
            # Multi-person violation
            if multi_person_result.get("violation", False):
                violations.append("multiple_persons")
                violation_details["multiple_persons"] = {
                    "detected": multi_person_result.get("num_people", 0),
                    "confidence": multi_person_result.get("confidence", 0),
                    "locations": multi_person_result.get("people_locations", [])
                }
            
            # Head pose violation
            if head_pose_result.get("violation", False):
                violations.append("head_pose")
                violation_details["head_pose"] = {
                    "direction": head_pose_result.get("direction", "unknown"),
                    "confidence": head_pose_result.get("confidence", 0)
                }
            
            # Body visibility violation
            if body_visibility_result.get("violation", False):
                violations.append("body_visibility")
                violation_details["body_visibility"] = {
                    "issue": body_visibility_result.get("visibility_status", "unknown"),
                    "confidence": body_visibility_result.get("confidence", 0)
                }
            
            results["violations"] = {
                "has_violations": len(violations) > 0,
                "violation_types": violations,
                "details": violation_details,
                "total_violations": len(violations)
            }
            
            # Summary
            results["summary"] = {
                "total_people": multi_person_result.get("num_people", 0),
                "head_direction": normalized_head_pose.get("direction", "unknown"),
                "body_visible": not body_visibility_result.get("violation", False),
                "overall_violation": len(violations) > 0,
                "models_used": {
                    "multi_person": multi_person_result.get("model_type", "unknown"),
                    "head_pose": "MobileNetV2" if "direction" in normalized_head_pose else "unknown",
                    "body_visibility": "Custom" if "visibility_status" in body_visibility_result else "unknown"
                }
            }
            
            print(f"✓ Unified inference completed - {len(violations)} violations detected")
            
        except Exception as e:
            print(f"✗ Error in unified inference: {e}")
            results["image_processed"] = False
            results["error"] = str(e)
        
        return results
    
    def get_model_status(self) -> Dict:
        """Get status of all models"""
        status = {
            "multi_person": {
                "custom_model_loaded": self._custom_model is not None and self._custom_model is not False,
                "fallback_model_loaded": self._fallback_model is not None,
                "currently_using": "custom" if self._custom_model is not None and self._custom_model is not False else "fallback",
                "last_error": self._last_error
            },
            "head_pose": {
                "available": True,
                "model_type": "MobileNetV2"
            },
            "body_visibility": {
                "available": True,
                "model_type": "Custom"
            },
            "unified_system": {
                "ready": self._model_loaded,
                "version": "1.0.0"
            }
        }
        
        return status


# Global instance
_unified_system = None

def get_unified_system() -> UnifiedDetectionSystem:
    """Get or create the unified detection system instance"""
    global _unified_system
    if _unified_system is None:
        _unified_system = UnifiedDetectionSystem()
    return _unified_system

def infer_multi_person_unified(image_bytes: bytes) -> Dict:
    """Enhanced multi-person detection (backward compatible)"""
    system = get_unified_system()
    return system.detect_multiple_persons(image_bytes)

def infer_unified(image_bytes: bytes) -> Dict:
    """Run all detections and return unified results"""
    system = get_unified_system()
    return system.unified_inference(image_bytes)

def get_system_status() -> Dict:
    """Get status of the unified detection system"""
    system = get_unified_system()
    return system.get_model_status()
