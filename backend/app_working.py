from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import logging
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy imports for models
_models_loaded = False
_model_loading_lock = threading.Lock()

def load_models_async():
    """Load models in background"""
    global _models_loaded
    with _model_loading_lock:
        if _models_loaded:
            return
        
        logger.info("Loading AI models in background...")
        try:
            from config import load_config
            from detections.head_pose import infer_head_pose
            from detections.multi_person import infer_multi_person
            from detections.body_visibility import infer_body_visibility
            from detections.unified_detection import infer_unified, get_system_status
            from alerts.twilio_client import send_whatsapp_alert
            from detections import head_pose_model as hpm
            
            # Store in globals for later use
            globals().update({
                'load_config': load_config,
                'infer_head_pose': infer_head_pose,
                'infer_multi_person': infer_multi_person,
                'infer_body_visibility': infer_body_visibility,
                'infer_unified': infer_unified,
                'get_system_status': get_system_status,
                'send_whatsapp_alert': send_whatsapp_alert,
                'hpm': hpm
            })
            
            _models_loaded = True
            logger.info("✓ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Load config immediately
    try:
        from config import load_config
        cfg = load_config()
        app.config.update(cfg)
        logger.info("✓ Configuration loaded")
    except Exception as e:
        logger.error(f"Config loading failed: {e}")

    # Start loading models in background
    threading.Thread(target=load_models_async, daemon=True).start()

    @app.get("/")
    def index():
        return jsonify({
            "message": "AI Exam Monitor API",
            "status": "running",
            "models_loaded": _models_loaded,
            "health": "/api/health",
            "endpoints": [
                "/api/health",
                "/api/detections/head_pose",
                "/api/detections/multi_person", 
                "/api/detections/body_visibility",
                "/api/detections/unified",
                "/api/system/status",
                "/api/alert",
            ],
        })

    @app.get("/api/health")
    def health():
        return {"status": "ok", "models_loaded": _models_loaded}

    def ensure_models_loaded():
        """Ensure models are loaded before processing"""
        if not _models_loaded:
            # Wait up to 30 seconds for models to load
            for i in range(30):
                if _models_loaded:
                    break
                time.sleep(1)
            
            if not _models_loaded:
                return jsonify({"error": "Models still loading, please try again in a moment"}), 503
        return None

    @app.post("/api/detections/head_pose")
    def head_pose_detection():
        error = ensure_models_loaded()
        if error:
            return error
            
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            image_bytes = base64.b64decode(image_b64.split(",")[-1])
        except Exception:
            return jsonify({"error": "invalid base64 image"}), 400

        result = globals()['infer_head_pose'](image_bytes)
        result.update({"user_id": user_id})
        return jsonify(result)

    @app.post("/api/detections/multi_person")
    def multi_person_detection():
        error = ensure_models_loaded()
        if error:
            return error
            
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            image_bytes = base64.b64decode(image_b64.split(",")[-1])
        except Exception:
            return jsonify({"error": "invalid base64 image"}), 400

        result = globals()['infer_multi_person'](image_bytes)
        result.update({"user_id": user_id})
        return jsonify(result)

    @app.post("/api/detections/body_visibility")
    def body_visibility_detection():
        error = ensure_models_loaded()
        if error:
            return error
            
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            image_bytes = base64.b64decode(image_b64.split(",")[-1])
        except Exception:
            return jsonify({"error": "invalid base64 image"}), 400

        result = globals()['infer_body_visibility'](image_bytes)
        result.update({"user_id": user_id})
        return jsonify(result)

    @app.post("/api/detections/unified")
    def unified_detection():
        """Run all detection models on the image and return comprehensive results"""
        error = ensure_models_loaded()
        if error:
            return error
            
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            image_bytes = base64.b64decode(image_b64.split(",")[-1])
        except Exception:
            return jsonify({"error": "invalid base64 image"}), 400

        result = globals()['infer_unified'](image_bytes)
        result.update({"user_id": user_id})
        return jsonify(result)

    @app.get("/api/system/status")
    def system_status():
        """Get comprehensive status of all detection models"""
        if not _models_loaded:
            return jsonify({
                "system": {"status": "loading", "models_loaded": False},
                "message": "Models are still loading..."
            })
            
        try:
            # Get unified system status
            unified_status = globals()['get_system_status']()
            hpm = globals()['hpm']
            
            # Add head pose model status
            head_pose_status = {
                "loaded": bool(getattr(hpm, "_model", None)),
                "model_path": getattr(hpm, "MODEL_PATH", None),
                "img_size": getattr(hpm, "_img_size", None),
                "classes": getattr(hpm, "_classes", None),
                "last_error": getattr(hpm, "_last_model_error", None)
            }
            
            # Combine all status information
            complete_status = {
                "system": {
                    "status": "operational",
                    "version": "2.0.0",
                    "models_loaded": 0,
                    "timestamp": ""
                },
                "models": {
                    "unified_detection": unified_status,
                    "head_pose_legacy": head_pose_status
                },
                "capabilities": {
                    "multi_person_detection": unified_status["multi_person"]["custom_model_loaded"] or unified_status["multi_person"]["fallback_model_loaded"],
                    "head_pose_detection": unified_status["head_pose"]["available"],
                    "body_visibility_detection": unified_status["body_visibility"]["available"],
                    "unified_inference": unified_status["unified_system"]["ready"]
                }
            }
            
            # Count loaded models
            models_loaded = 0
            if unified_status["multi_person"]["custom_model_loaded"] or unified_status["multi_person"]["fallback_model_loaded"]:
                models_loaded += 1
            if unified_status["head_pose"]["available"]:
                models_loaded += 1
            if unified_status["body_visibility"]["available"]:
                models_loaded += 1
            
            complete_status["system"]["models_loaded"] = models_loaded
            
            import datetime
            complete_status["system"]["timestamp"] = datetime.datetime.now().isoformat()
            
            return jsonify(complete_status)
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return jsonify({
                "system": {"status": "error", "error": str(e)},
                "models": {},
                "capabilities": {}
            }), 500

    @app.get("/api/debug/model")
    def debug_model():
        if not _models_loaded:
            return jsonify({"loaded": False, "status": "loading"})
            
        # report whether trained model is loaded
        try:
            hpm = globals()['hpm']
            loaded = bool(getattr(hpm, "_model", None))
            return jsonify({
                "loaded": loaded,
                "model_path": getattr(hpm, "MODEL_PATH", None),
                "img_size": getattr(hpm, "_img_size", None),
                "classes": getattr(hpm, "_classes", None),
                "last_error": getattr(hpm, "_last_model_error", None),
            })
        except Exception as e:
            return jsonify({"loaded": False, "error": str(e)}), 500

    @app.post("/api/alert")
    def send_alert():
        error = ensure_models_loaded()
        if error:
            return error
            
        payload = request.get_json(silent=True) or {}
        student = payload.get("student")
        violation = payload.get("violation")
        timestamp = payload.get("timestamp")

        if not (student and violation):
            return jsonify({"error": "student and violation are required"}), 400

        try:
            send_whatsapp_alert = globals()['send_whatsapp_alert']
            ok, info = send_whatsapp_alert(
                to_phone=os.environ.get("TWILIO_WHATSAPP_TO", ""),
                body=f"[Exam Alert] {student}: {violation} @ {timestamp}",
            )
            return (jsonify({"sent": True, "sid": info}) if ok else
                    (jsonify({"sent": False, "error": info}), 500))
        except Exception as e:
            return jsonify({"sent": False, "error": f"Alert system not loaded: {e}"}), 500

    return app

if __name__ == "__main__":
    # Set environment variables for models
    os.environ['HEAD_POSE_MODEL_PATH'] = r"C:\Users\sanke\OneDrive\Documents\EDI\Project\Project\ml\models\head_pose_mobilenet_final.h5"
    os.environ['HEAD_POSE_MODEL_INFO_PATH'] = r"C:\Users\sanke\OneDrive\Documents\EDI\Project\Project\ml\models\head_pose_mobilenet_info.json"
    
    application = create_app()
    port = int(os.environ.get("PORT", "5000"))
    
    logger.info(f"🚀 Starting AI Exam Monitor on http://127.0.0.1:{port}")
    logger.info("📍 Models will load in background - API available immediately")
    logger.info("🔗 Frontend: Open C:\\Users\\sanke\\OneDrive\\Documents\\EDI\\Project\\Project\\frontend\\index.html")
    
    application.run(host="0.0.0.0", port=port, debug=False, threaded=True)
