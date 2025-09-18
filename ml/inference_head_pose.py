import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json
import argparse

class HeadPoseInference:
    def __init__(self, model_path, model_info_path=None):
        """
        Initialize head pose inference
        
        Args:
            model_path: Path to the trained Keras model (.h5 file)
            model_info_path: Path to model info JSON file (optional)
        """
        self.model = keras.models.load_model(model_path)
        
        # Load model info if available
        if model_info_path and os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
            self.class_names = self.model_info['classes']
            self.img_size = self.model_info['img_size']
        else:
            # Default values
            self.class_names = ["forward", "left", "right", "down", "up"]
            self.img_size = 160
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.class_names}")
        print(f"Image size: {self.img_size}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image
        image = image.resize((self.img_size, self.img_size))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_tensor = np.expand_dims(img_array, axis=0)
        
        return img_tensor
    
    def detect_face(self, image):
        """
        Detect face in image using OpenCV
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Cropped face image or None if no face detected
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            
            # Add some padding around the face
            padding = 0.2
            x_start = max(0, int(x - padding * w))
            y_start = max(0, int(y - padding * h))
            x_end = min(image.shape[1], int(x + w + padding * w))
            y_end = min(image.shape[0], int(y + h + padding * h))
            
            # Crop face
            face_crop = image[y_start:y_end, x_start:x_end]
            return face_crop
        
        return None
    
    def predict_pose(self, image, detect_face=True):
        """
        Predict head pose from image
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            detect_face: Whether to detect and crop face first
            
        Returns:
            Dictionary with pose prediction results
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Detect and crop face if requested
        if detect_face:
            face_crop = self.detect_face(image)
            if face_crop is None:
                return {
                    "pose": "unknown",
                    "confidence": 0.0,
                    "probabilities": {},
                    "face_detected": False
                }
            image = face_crop
        
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(img_tensor, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(probabilities)
        predicted_pose = self.class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Create probabilities dictionary
        prob_dict = {self.class_names[i]: float(probabilities[i]) for i in range(len(self.class_names))}
        
        return {
            "pose": predicted_pose,
            "confidence": confidence,
            "probabilities": prob_dict,
            "face_detected": True
        }
    
    def predict_batch(self, image_paths):
        """
        Predict head pose for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict_pose(path)
                result["image_path"] = path
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": path,
                    "pose": "error",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Head pose inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.h5 file)")
    parser.add_argument("--model_info", help="Path to model info JSON file")
    parser.add_argument("--image", help="Path to single image for prediction")
    parser.add_argument("--image_dir", help="Directory containing images for batch prediction")
    parser.add_argument("--output", help="Output file for batch predictions (JSON)")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = HeadPoseInference(args.model_path, args.model_info)
    
    if args.image:
        # Single image prediction
        print(f"Predicting pose for: {args.image}")
        result = inference.predict_pose(args.image)
        print(f"Predicted pose: {result['pose']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"All probabilities: {result['probabilities']}")
    
    elif args.image_dir:
        # Batch prediction
        print(f"Predicting poses for images in: {args.image_dir}")
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = []
        for file in os.listdir(args.image_dir):
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(args.image_dir, file))
        
        if not image_paths:
            print("No image files found in directory!")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Predict poses
        results = inference.predict_batch(image_paths)
        
        # Print results
        for result in results:
            print(f"\n{result['image_path']}:")
            print(f"  Pose: {result['pose']}")
            print(f"  Confidence: {result['confidence']:.4f}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    else:
        print("Please specify either --image or --image_dir")

if __name__ == "__main__":
    main()
