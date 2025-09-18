from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os
import sys
import cv2
import numpy as np
from PIL import Image
import io

print("🚀 Starting AI Exam Monitor - Fixed Version")

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    print("✓ Flask app created with CORS enabled")

    @app.get("/")
    def index():
        return jsonify({
            "message": "AI Exam Monitor API - Fixed Version",
            "status": "running",
            "version": "3.0",
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
        return {"status": "ok", "message": "All systems operational"}

    def process_image_bytes(image_b64):
        """Convert base64 to OpenCV image"""
        try:
            # Remove data:image/jpeg;base64, prefix if present
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_b64)
            
            # Convert to PIL image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image, None
        except Exception as e:
            return None, str(e)

    # Simple head pose detection using OpenCV
    def detect_head_pose_simple(image):
        """Simple head pose detection using face detection"""
        try:
            # Load cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {
                    "head_pose": "not_detected",
                    "confidence": 0.0,
                    "violation": False,
                    "method": "opencv_simple"
                }
            
            # Use the largest face
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            
            # Simple heuristic for head pose based on face position
            img_center_x = image.shape[1] // 2
            face_center_x = x + w // 2
            
            # Determine head direction
            if face_center_x < img_center_x - w * 0.3:
                head_pose = "right"
                violation = True
            elif face_center_x > img_center_x + w * 0.3:
                head_pose = "left"
                violation = True
            else:
                head_pose = "forward"
                violation = False
            
            return {
                "head_pose": head_pose,
                "confidence": 0.8,
                "violation": violation,
                "method": "opencv_simple",
                "face_location": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            }
            
        except Exception as e:
            return {
                "head_pose": "error",
                "confidence": 0.0,
                "violation": False,
                "error": str(e),
                "method": "opencv_simple"
            }

    # Simple multi-person detection using OpenCV
    def detect_multi_person_simple(image):
        """Simple multi-person detection using HOG"""
        try:
            # Initialize HOG descriptor
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect people
            boxes, weights = hog.detectMultiScale(image, winStride=(8,8))
            
            # Filter detections by weight
            filtered_boxes = []
            for i, box in enumerate(boxes):
                if weights[i] > 0.5:  # Confidence threshold
                    filtered_boxes.append(box)
            
            num_people = len(filtered_boxes)
            
            return {
                "num_people": num_people,
                "confidence": max(weights) if len(weights) > 0 else 0.0,
                "violation": num_people > 1,
                "method": "opencv_hog",
                "people_locations": [
                    {"x": int(x), "y": int(y), "width": int(w), "height": int(h)} 
                    for x, y, w, h in filtered_boxes
                ]
            }
            
        except Exception as e:
            return {
                "num_people": 0,
                "confidence": 0.0,
                "violation": False,
                "error": str(e),
                "method": "opencv_hog"
            }

    # Simple body visibility detection
    def detect_body_visibility_simple(image):
        """Simple body visibility detection using face detection"""
        try:
            # Load cascade classifiers
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {
                    "body_visible": False,
                    "face_detected": False,
                    "confidence": 0.0,
                    "violation": True,
                    "method": "opencv_simple"
                }
            
            # Check if face is in upper portion of image
            x, y, w, h = faces[0]  # Use first face
            img_height = image.shape[0]
            
            # Face should be in upper 60% of image for good body visibility
            face_in_upper_portion = (y + h) < (img_height * 0.6)
            
            # Check face size relative to image
            face_area = w * h
            img_area = image.shape[0] * image.shape[1]
            face_ratio = face_area / img_area
            
            # Good visibility if face is reasonable size and in upper portion
            good_visibility = face_in_upper_portion and (0.01 < face_ratio < 0.3)
            
            return {
                "body_visible": good_visibility,
                "face_detected": True,
                "confidence": 0.8 if good_visibility else 0.4,
                "violation": not good_visibility,
                "method": "opencv_simple",
                "face_location": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "face_ratio": float(face_ratio)
            }
            
        except Exception as e:
            return {
                "body_visible": False,
                "face_detected": False,
                "confidence": 0.0,
                "violation": True,
                "error": str(e),
                "method": "opencv_simple"
            }

    @app.post("/api/detections/head_pose")
    def head_pose_detection():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400
        
        # Process image
        image, error = process_image_bytes(image_b64)
        if image is None:
            return jsonify({"error": f"Image processing failed: {error}"}), 400
        
        # Detect head pose
        result = detect_head_pose_simple(image)
        result["user_id"] = user_id
        
        return jsonify(result)

    @app.post("/api/detections/multi_person")
    def multi_person_detection():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400
        
        # Process image
        image, error = process_image_bytes(image_b64)
        if image is None:
            return jsonify({"error": f"Image processing failed: {error}"}), 400
        
        # Detect multiple people
        result = detect_multi_person_simple(image)
        result["user_id"] = user_id
        
        return jsonify(result)

    @app.post("/api/detections/body_visibility")
    def body_visibility_detection():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400
        
        # Process image
        image, error = process_image_bytes(image_b64)
        if image is None:
            return jsonify({"error": f"Image processing failed: {error}"}), 400
        
        # Detect body visibility
        result = detect_body_visibility_simple(image)
        result["user_id"] = user_id
        
        return jsonify(result)

    @app.post("/api/detections/unified")
    def unified_detection():
        """Run all detection models on the image and return comprehensive results"""
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("image_b64")
        user_id = payload.get("user_id")
        
        if not image_b64:
            return jsonify({"error": "image_b64 required"}), 400
        
        # Process image
        image, error = process_image_bytes(image_b64)
        if image is None:
            return jsonify({"error": f"Image processing failed: {error}"}), 400
        
        # Run all detections
        head_pose_result = detect_head_pose_simple(image)
        multi_person_result = detect_multi_person_simple(image)
        body_visibility_result = detect_body_visibility_simple(image)
        
        # Combine results
        unified_result = {
            "user_id": user_id,
            "timestamp": "",
            "head_pose": head_pose_result,
            "multi_person": multi_person_result,
            "body_visibility": body_visibility_result,
            "overall_violation": (
                head_pose_result.get("violation", False) or
                multi_person_result.get("violation", False) or
                body_visibility_result.get("violation", False)
            )
        }
        
        import datetime
        unified_result["timestamp"] = datetime.datetime.now().isoformat()
        
        return jsonify(unified_result)

    @app.get("/api/system/status")
    def system_status():
        """Get system status"""
        return jsonify({
            "system": {
                "status": "operational",
                "version": "3.0-opencv",
                "models_loaded": 3,
                "timestamp": ""
            },
            "models": {
                "head_pose": {"available": True, "method": "opencv_simple"},
                "multi_person": {"available": True, "method": "opencv_hog"},
                "body_visibility": {"available": True, "method": "opencv_simple"}
            },
            "capabilities": {
                "multi_person_detection": True,
                "head_pose_detection": True,
                "body_visibility_detection": True,
                "unified_inference": True
            }
        })

    @app.post("/api/alert")
    def send_alert():
        payload = request.get_json(silent=True) or {}
        student = payload.get("student")
        violation = payload.get("violation")
        timestamp = payload.get("timestamp")

        if not (student and violation):
            return jsonify({"error": "student and violation are required"}), 400

        # Simple console alert (Twilio can be added later)
        alert_message = f"[Exam Alert] {student}: {violation} @ {timestamp}"
        print(f"🚨 ALERT: {alert_message}")
        
        return jsonify({"sent": True, "message": alert_message, "method": "console"})

    return app

if __name__ == "__main__":
    print("🔧 Creating Flask application...")
    
    application = create_app()
    port = int(os.environ.get("PORT", "5000"))
    
    print(f"✅ AI Exam Monitor Fixed Version Ready!")
    print(f"🌐 Server: http://127.0.0.1:{port}")
    print(f"🔗 Frontend: C:\\Users\\sanke\\OneDrive\\Documents\\EDI\\Project\\Project\\frontend\\index.html")
    print("📊 All detection models: OpenCV-based (fast startup)")
    print("🚀 Starting server...")
    
    try:
        application.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)
