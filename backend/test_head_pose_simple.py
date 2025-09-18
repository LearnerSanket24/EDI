import cv2
import numpy as np
import base64
import io
from PIL import Image
import requests

def create_test_image():
    """Create a simple test image with a face"""
    # Create a blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Draw a simple face
    center_x, center_y = 320, 240
    
    # Face outline (circle)
    cv2.circle(img, (center_x, center_y), 80, (255, 220, 180), -1)  # Skin color face
    cv2.circle(img, (center_x, center_y), 80, (200, 180, 150), 3)   # Face border
    
    # Eyes
    cv2.circle(img, (center_x-25, center_y-20), 8, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (center_x+25, center_y-20), 8, (50, 50, 50), -1)  # Right eye
    
    # Nose
    cv2.circle(img, (center_x, center_y), 3, (180, 150, 120), -1)
    
    # Mouth
    cv2.ellipse(img, (center_x, center_y+25), (15, 8), 0, 0, 180, (100, 50, 50), 2)
    
    return img

def image_to_base64(img):
    """Convert OpenCV image to base64 string"""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Convert to bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    
    # Encode to base64
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_b64}"

def test_head_pose():
    """Test the head pose detection endpoint"""
    print("🧪 Testing Head Pose Detection")
    print("=" * 40)
    
    # Create test image
    test_img = create_test_image()
    img_b64 = image_to_base64(test_img)
    
    # Test data
    test_data = {
        "image_b64": img_b64,
        "user_id": "test_user"
    }
    
    try:
        # Send request to backend
        response = requests.post(
            "http://127.0.0.1:5000/api/detections/head_pose",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Head Pose Detection: SUCCESS")
            print(f"   Pose: {result.get('pose', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Violation: {result.get('violation', False)}")
            print(f"   Method: {result.get('method', 'unknown')}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_all_endpoints():
    """Test all detection endpoints"""
    print("🧪 Testing All Detection Endpoints")
    print("=" * 40)
    
    # Create test image
    test_img = create_test_image()
    img_b64 = image_to_base64(test_img)
    
    endpoints = [
        ("Head Pose", "/api/detections/head_pose"),
        ("Multi Person", "/api/detections/multi_person"),
        ("Body Visibility", "/api/detections/body_visibility"),
        ("Unified", "/api/detections/unified")
    ]
    
    test_data = {
        "image_b64": img_b64,
        "user_id": "test_user"
    }
    
    for name, endpoint in endpoints:
        try:
            response = requests.post(
                f"http://127.0.0.1:5000{endpoint}",
                json=test_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {name}: SUCCESS")
                
                # Print relevant fields
                if 'pose' in result:
                    print(f"   Pose: {result['pose']}")
                if 'num_people' in result:
                    print(f"   People: {result['num_people']}")
                if 'confidence' in result:
                    print(f"   Confidence: {result['confidence']:.2f}")
                if 'violation' in result:
                    print(f"   Violation: {result['violation']}")
                    
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {name}: {e}")

if __name__ == "__main__":
    # Test individual head pose
    test_head_pose()
    print()
    
    # Test all endpoints
    test_all_endpoints()
    
    print("\n🎯 Test completed!")
    print("If head pose shows 'forward' with good confidence, the system is working!")
