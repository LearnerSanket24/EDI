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
except Exception:  # pragma: no cover
    YOLO = None


_yolo = None


def _get_model():
    global _yolo
    if _yolo is None and YOLO is not None:
        # nano model is small and CPU-friendly
        _yolo = YOLO('yolov8n.pt')
    return _yolo


def _bytes_to_numpy(image_bytes: bytes):
    if Image is None:
        return None
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if np is None:
        return None
    return np.array(img)


def infer_multi_person_yolov8(image_bytes: bytes) -> Dict:
    model = _get_model()
    if model is None:
        return {"num_people": 0, "confidence": 0.0, "violation": False, "using_yolo": False}

    frame = _bytes_to_numpy(image_bytes)
    if frame is None:
        return {"num_people": 0, "confidence": 0.0, "violation": False, "using_yolo": False}
    results = model.predict(source=frame, classes=[0], conf=0.25, verbose=False)  # class 0 = person in COCO
    if not results:
        return {"num_people": 0, "confidence": 0.0, "violation": False, "using_yolo": True}

    r = results[0]
    num_people = int(len(r.boxes))
    conf = float(r.boxes.conf.max().item()) if num_people > 0 else 0.0
    violation = num_people > 1
    return {"num_people": num_people, "confidence": conf, "violation": violation, "using_yolo": True}


