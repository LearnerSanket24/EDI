import io
import os
import json
from typing import Dict, Optional



MODEL_PATH = os.environ.get(
    'HEAD_POSE_MODEL_PATH',
    os.path.join(os.path.dirname(__file__), '..', 'models', 'head_pose_mobilenet_final.h5')
)
MODEL_INFO_PATH = os.environ.get(
    'HEAD_POSE_MODEL_INFO_PATH',
    os.path.join(os.path.dirname(__file__), '..', 'models', 'head_pose_mobilenet_info.json')
)
CLASSES_DEFAULT = ["forward", "left", "right", "down", "up"]


_model = None
_img_size = 160
_classes = CLASSES_DEFAULT
_last_model_error: str | None = None


def _load_model_if_available():
    global _model, _img_size, _classes, _last_model_error
    print(f"Loading model from: {MODEL_PATH}")
    
    # Check if model is already loaded
    if _model is not None:
        print("Model already loaded")
        return
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        # Try to find model in alternative locations
        alt_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'head_pose_mobilenet_final.h5'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'models', 'head_pose_mobilenet_final.h5'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'head_pose_mobilenet_best.h5')
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"Found model at alternative path: {alt_path}")
                globals()['MODEL_PATH'] = alt_path
                break
        
        # If still not found, report error
        if not os.path.exists(MODEL_PATH):
            _last_model_error = f"Model file not found at {MODEL_PATH} or any alternative locations"
            print(f"Error: {_last_model_error}")
            return
    
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    
    try:
        import tensorflow as tf
        print("TensorFlow imported successfully")
        from tensorflow import keras
        
        # Load model with error handling
        try:
            _model = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            _last_model_error = f"Error loading model: {str(e)}\n{traceback.format_exc()[:500]}"
            return

        # Load model info if available
        model_info_path = MODEL_INFO_PATH
        if not os.path.exists(model_info_path):
            # Try to find info in same directory as model
            model_info_path = MODEL_PATH.replace('.h5', '_info.json')
        
        if os.path.exists(model_info_path):
            try:
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                _classes = model_info.get('classes', CLASSES_DEFAULT)
                _img_size = model_info.get('img_size', 160)
                print(f"Loaded model info: classes={_classes}, img_size={_img_size}")
            except Exception as e:
                print(f"Error loading model info: {str(e)}")
                _classes = CLASSES_DEFAULT
                _img_size = 160
        else:
            _classes = CLASSES_DEFAULT
            _img_size = 160
            print(f"Using default model info: classes={_classes}, img_size={_img_size}")

        _last_model_error = None
    except Exception as e:
        # If tensorflow/keras not available, keep _model as None to disable trained path
        _model = None
        import traceback
        _last_model_error = f"Error in model loading process: {str(e)}\n{traceback.format_exc()[:500]}"
        print(f"Exception during model loading: {str(e)}")


def _preprocess(image_bytes: bytes):
    # Decode
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return None

    # Convert to OpenCV for face detection
    try:
        import numpy as np
        import cv2
        cv_img = np.array(img)[:, :, ::-1]
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    except Exception:
        # If numpy/opencv unavailable, skip face cropping
        faces = []
        import numpy as np  # ensure cv_img exists for later conversion without cv2
        cv_img = np.array(img)[:, :, ::-1]
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        x0 = max(0, x - int(0.15 * w))
        y0 = max(0, y - int(0.15 * h))
        x1 = min(cv_img.shape[1], x + w + int(0.15 * w))
        y1 = min(cv_img.shape[0], y + h + int(0.15 * h))
        cv_img = cv_img[y0:y1, x0:x1]

    # Back to PIL
    try:
        import cv2
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    except Exception:
        from PIL import Image
        img = Image.fromarray(cv_img[:, :, ::-1])

    try:
        import numpy as np
        # Resize image
        img = img.resize((_img_size, _img_size))

        # Convert to array and normalize
        img_array = np.array(img) / 255.0

        # Add batch dimension
        img_tensor = np.expand_dims(img_array, axis=0)

        return img_tensor
    except Exception:
        return None


def infer_head_pose_trained(image_bytes: bytes) -> Dict:
    _load_model_if_available()
    if _model is None:
        return {"direction": "forward", "head_pose": "forward", "confidence": 0.0, "violation": False, "using_trained": False, "model_error": _last_model_error}
    inp = _preprocess(image_bytes)
    if inp is None:
        return {"direction": "unknown", "head_pose": "unknown", "confidence": 0.0, "violation": False, "using_trained": False, "model_error": _last_model_error}
    try:
        import numpy as np
        # Make prediction
        predictions = _model.predict(inp, verbose=0)
        probabilities = predictions[0]

        # Get predicted class and confidence
        predicted_class_idx = np.argmax(probabilities)
        predicted_direction = _classes[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])

        # Consider it a violation if looking away with high confidence
        violation = predicted_direction in ("left", "right", "down") and confidence >= 0.6
        
        return {
            "direction": predicted_direction,
            "head_pose": predicted_direction,
            "confidence": confidence,
            "violation": violation,
            "using_trained": True,
            "model_type": "MobileNet H5"
        }
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()[:1000]
        return {
            "direction": "unknown",
            "head_pose": "unknown",
            "confidence": 0.0,
            "violation": False,
            "using_trained": False,
            "error": error_msg
        }


