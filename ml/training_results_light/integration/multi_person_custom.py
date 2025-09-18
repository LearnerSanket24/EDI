from typing import Dict
import base64
import io
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import numpy as np
except Exception:
    np = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

_custom_model = None
_fallback_model = None

def _get_custom_model():
    global _custom_model
    if _custom_model is None and YOLO is not None:
        try:
            # Try to load custom trained model first
            _custom_model = YOLO('models/crowdhuman_custom.pt')
            print("Loaded custom CrowdHuman model")
        except Exception as e:
            print(f"Could not load custom model: {e}")
            _custom_model = False
    return _custom_model

def _get_fallback_model():
    global _fallback_model
    if _fallback_model is None and YOLO is not None:
        # Fallback to pre-trained YOLOv8n
        _fallback_model = YOLO('yolov8n.pt')
        print("Loaded fallback YOLOv8n model")
    return _fallback_model

def _bytes_to_numpy(image_bytes: bytes):
    if Image is None:
        return None
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if np is None:
        return None
    return np.array(img)

def infer_multi_person_custom(image_bytes: bytes) -> Dict:
    """Multi-person detection using custom trained CrowdHuman model with fallback"""
    
    # Try custom model first
    model = _get_custom_model()
    using_custom = True
    
    # Fallback to pre-trained model if custom fails
    if model is False or model is None:
        model = _get_fallback_model()
        using_custom = False
        if model is None:
            return {"num_people": 0, "confidence": 0.0, "violation": False, "using_custom": False, "error": "No models available"}

    frame = _bytes_to_numpy(image_bytes)
    if frame is None:
        return {"num_people": 0, "confidence": 0.0, "violation": False, "using_custom": using_custom, "error": "Could not process image"}
    
    try:
        # Run inference with appropriate confidence threshold
        conf_threshold = 0.3 if using_custom else 0.25  # Lower threshold for custom model
        results = model.predict(source=frame, classes=[0], conf=conf_threshold, verbose=False)
        
        if not results:
            return {"num_people": 0, "confidence": 0.0, "violation": False, "using_custom": using_custom}

        r = results[0]
        num_people = int(len(r.boxes)) if r.boxes is not None else 0
        conf = float(r.boxes.conf.max().item()) if num_people > 0 else 0.0
        violation = num_people > 1
        
        return {
            "num_people": num_people, 
            "confidence": conf, 
            "violation": violation, 
            "using_custom": using_custom,
            "model_type": "CrowdHuman Custom" if using_custom else "YOLOv8n Fallback"
        }
        
    except Exception as e:
        return {"num_people": 0, "confidence": 0.0, "violation": False, "using_custom": using_custom, "error": str(e)}
