from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os

from config import load_config
from detections.head_pose import infer_head_pose
from detections.multi_person import infer_multi_person
from detections.body_visibility import infer_body_visibility
from detections.unified_detection import infer_unified, get_system_status
from alerts.twilio_client import send_whatsapp_alert
from detections import head_pose_model as hpm


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    cfg = load_config()
    app.config.update(cfg)

    # Background warmup to avoid first-request stalls
    def _background_warmup():
        try:
            from detections.unified_detection import get_unified_system
            from detections import head_pose_model as hpm
            import io
            from PIL import Image
            sys = get_unified_system()
            # Tiny black image to trigger model init
            img = Image.new('RGB', (64, 64), (0, 0, 0))
            buf = io.BytesIO(); img.save(buf, format='JPEG'); img_bytes = buf.getvalue()
            # Load head pose model if available
            hpm._load_model_if_available()
            # Trigger multi-person model load
            sys.detect_multiple_persons(img_bytes)
            print("Warmup complete")
        except Exception as e:
            print("Warmup failed:", str(e))
    try:
        import threading
        threading.Thread(target=_background_warmup, daemon=True).start()
    except Exception as _:
        pass

    # Load models on startup to check for errors
    print("Starting model loading...")
    try:
        from detections.unified_detection import get_system_status
        status = get_system_status()
        print("Model status:", status)
    except Exception as e:
        print("Model loading error on startup:", str(e))

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/warmup")
    def api_warmup():
        try:
            import threading
            threading.Thread(target=_background_warmup, daemon=True).start()
            return {"started": True}
        except Exception as e:
            return {"started": False, "error": str(e)}, 500

    @app.get("/api")
    def api_index():
        return jsonify({
            "message": "AI Exam Monitor API",
            "health": "/api/health",
            "endpoints": [
                "/api/detections/head_pose",
                "/api/detections/multi_person",
                "/api/detections/body_visibility",
                "/api/detections/unified",
                "/api/system/status",
                "/api/alert",
            ],
            "frontend": "Static frontend served at /",
        })

    @app.post("/api/detections/head_pose")
    def head_pose_detection():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            image_bytes = base64.b64decode(image_b64.split(",")[-1])
        except Exception:
            return jsonify({"error": "invalid base64 image"}), 400

        result = infer_head_pose(image_bytes)
        result.update({"user_id": user_id})
        return jsonify(result)

    @app.get("/api/debug/model")
    def debug_model():
        # report whether trained model is loaded
        try:
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

    @app.post("/api/detections/multi_person")
    def multi_person_detection():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            image_bytes = base64.b64decode(image_b64.split(",")[-1])
        except Exception:
            return jsonify({"error": "invalid base64 image"}), 400

        result = infer_multi_person(image_bytes)
        result.update({"user_id": user_id})
        return jsonify(result)

    @app.post("/api/detections/body_visibility")
    def body_visibility_detection():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            image_bytes = base64.b64decode(image_b64.split(",")[-1])
        except Exception:
            return jsonify({"error": "invalid base64 image"}), 400

        result = infer_body_visibility(image_bytes)
        result.update({"user_id": user_id})
        return jsonify(result)

    @app.post("/api/alert")
    def send_alert():
        payload = request.get_json(silent=True) or {}
        student = payload.get("student")
        violation = payload.get("violation")
        timestamp = payload.get("timestamp")

        if not (student and violation):
            return jsonify({"error": "student and violation are required"}), 400

        ok, info = send_whatsapp_alert(
            to_phone=os.environ.get("TWILIO_WHATSAPP_TO", ""),
            body=f"[Exam Alert] {student}: {violation} @ {timestamp}",
        )
        return (jsonify({"sent": True, "sid": info}) if ok else
                (jsonify({"sent": False, "error": info}), 500))

    @app.post("/api/detections/unified")
    def unified_detection():
        """Run all detection models on the image and return comprehensive results"""
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400

        try:
            image_bytes = base64.b64decode(image_b64.split(",")[-1])
        except Exception:
            return jsonify({"error": "invalid base64 image"}), 400

        result = infer_unified(image_bytes)
        result.update({"user_id": user_id})
        return jsonify(result)

    @app.get("/api/system/status")
    def system_status():
        """Get comprehensive status of all detection models"""
        try:
            # Get unified system status
            unified_status = get_system_status()
            
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
            return jsonify({
                "system": {"status": "error", "error": str(e)},
                "models": {},
                "capabilities": {}
            }), 500

    # Serve static frontend
    @app.route("/", defaults={"path": "index.html"})
    @app.route("/<path:path>")
    def serve_frontend(path):
        from os import path as osp

        project_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
        # Candidate frontend directories in priority order (new frontend first)
        candidate_dirs = [
            osp.join(project_root, "backend", "static"),
            osp.join(project_root, "static"),
            osp.join(project_root, "backend", "Safe_Exam_Monitor", "my-app", "dist"),
            osp.join(project_root, "Safe_Exam_Monitor", "my-app", "dist"),
            osp.join(project_root, "frontend"),
        ]

        def try_serve_file(base_dir: str, rel_path: str):
            file_path_local = osp.join(base_dir, rel_path)
            if osp.exists(file_path_local) and not osp.isdir(file_path_local):
                with open(file_path_local, "rb") as f:
                    content_local = f.read()
                mime_types_local = {
                    ".html": "text/html",
                    ".css": "text/css",
                    ".js": "application/javascript",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".ico": "image/x-icon",
                    ".svg": "image/svg+xml",
                    ".json": "application/json",
                }
                ext_local = osp.splitext(rel_path)[1].lower()
                content_type_local = mime_types_local.get(ext_local, "application/octet-stream")
                return content_local, 200, {"Content-Type": content_type_local}
            return None

        # Serve static assets (not API routes)
        if path != "index.html" and not path.startswith("api/"):
            for base in candidate_dirs:
                if osp.exists(base):
                    # Try direct path
                    served = try_serve_file(base, path)
                    if served:
                        return served
                    # Try assets subfolder (for Vite builds)
                    served = try_serve_file(base, osp.join("assets", path))
                    if served:
                        return served

        # Default to index.html for SPA-like routing
        for base in candidate_dirs:
            index_path = osp.join(base, "index.html")
            if osp.exists(index_path):
                with open(index_path, "rb") as f:
                    return f.read(), 200, {"Content-Type": "text/html"}

        return "Frontend not found", 404

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.environ.get("PORT", "5000"))
    application.run(host="0.0.0.0", port=port, debug=True)


