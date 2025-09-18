from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import os

def create_minimal_app():
    app = Flask(__name__)
    CORS(app)

    @app.get("/")
    def index():
        return jsonify({
            "message": "AI Exam Monitor API - Minimal Version",
            "status": "running",
            "health": "/api/health",
            "endpoints": [
                "/api/health",
                "/api/test"
            ],
        })

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "minimal"}

    @app.get("/api/test")
    def test():
        return {"test": "successful", "message": "Basic endpoint working"}

    return app

if __name__ == "__main__":
    app = create_minimal_app()
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting minimal Flask app on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
