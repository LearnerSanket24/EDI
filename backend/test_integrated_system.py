#!/usr/bin/env python3
"""
Comprehensive Testing Script for Integrated Detection System
Tests the CrowdHuman multi-person detection model and head pose model integration
"""

import os
import base64
import json
import time
import requests
from pathlib import Path
from typing import Dict, List
import argparse

class IntegratedSystemTester:
    """Test the integrated detection system"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.test_results = {
            "system_status": {},
            "individual_tests": {},
            "unified_tests": {},
            "performance_metrics": {},
            "timestamp": ""
        }
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def test_system_status(self) -> Dict:
        """Test system status endpoint"""
        print("[Status] Testing system status...")
        
        try:
            response = requests.get(f"{self.base_url}/api/system/status", timeout=30)
            
            if response.status_code == 200:
                status_data = response.json()
                
                print(f"[Status] System Status: {status_data['system']['status']}")
                print(f"[Version] Version: {status_data['system']['version']}")
                print(f"[Models] Models Loaded: {status_data['system']['models_loaded']}")
                
                # Check individual capabilities
                caps = status_data['capabilities']
                print(f"[OK] Multi-person Detection: {'[OK]' if caps['multi_person_detection'] else '[ERR]'}")
                print(f"[OK] Head Pose Detection: {'[OK]' if caps['head_pose_detection'] else '[ERR]'}")
                print(f"[OK] Body Visibility Detection: {'[OK]' if caps['body_visibility_detection'] else '[ERR]'}")
                print(f"[OK] Unified Inference: {'[OK]' if caps['unified_inference'] else '[ERR]'}")
                
                # Check which multi-person model is being used
                mp_model = status_data['models']['unified_detection']['multi_person']
                model_type = "Custom CrowdHuman" if mp_model['custom_model_loaded'] else "YOLOv8n Fallback"
                print(f"[Model] Multi-person Model: {model_type}")
                
                return {"success": True, "data": status_data}
                
            else:
                print(f"[Error] System status check failed: {response.status_code}")
                return {"success": False, "error": f"Status code: {response.status_code}"}
                
        except Exception as e:
            print(f"[Error] Error checking system status: {e}")
            return {"success": False, "error": str(e)}
    
    def test_individual_endpoint(self, endpoint: str, image_path: str, test_name: str) -> Dict:
        """Test individual detection endpoint"""
        print(f"[Test] Testing {test_name}...")
        
        try:
            if not os.path.exists(image_path):
                print(f"[Error] Test image not found: {image_path}")
                return {"success": False, "error": "Image not found"}
            
            image_b64 = self.encode_image(image_path)
            if not image_b64:
                return {"success": False, "error": "Could not encode image"}
            
            payload = {
                "image_b64": f"data:image/jpeg;base64,{image_b64}",
                "user_id": f"test_user_{test_name}"
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}{endpoint}", 
                                   json=payload, 
                                   timeout=60)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                
                print(f"[OK] {test_name} completed in {processing_time:.2f}s")
                
                # Print relevant results based on endpoint
                if "multi_person" in endpoint:
                    num_people = result_data.get('num_people', 0)
                    confidence = result_data.get('confidence', 0)
                    violation = result_data.get('violation', False)
                    model_type = result_data.get('model_type', 'unknown')
                    
                    print(f"  [People] People detected: {num_people}")
                    print(f"  [Conf] Confidence: {confidence:.3f}")
                    print(f"  [Viol] Violation: {'Yes' if violation else 'No'}")
                    print(f"  [Model] Model: {model_type}")
                    
                elif "head_pose" in endpoint:
                    direction = result_data.get('direction', 'unknown')
                    confidence = result_data.get('confidence', 0)
                    violation = result_data.get('violation', False)
                    
                    print(f"  [Head] Head direction: {direction}")
                    print(f"  [Conf] Confidence: {confidence:.3f}")
                    print(f"  [Viol] Violation: {'Yes' if violation else 'No'}")
                
                elif "body_visibility" in endpoint:
                    status = result_data.get('visibility_status', 'unknown')
                    confidence = result_data.get('confidence', 0)
                    violation = result_data.get('violation', False)
                    
                    print(f"  [Vis] Visibility: {status}")
                    print(f"  [Conf] Confidence: {confidence:.3f}")
                    print(f"  [Viol] Violation: {'Yes' if violation else 'No'}")
                
                return {
                    "success": True,
                    "data": result_data,
                    "processing_time": processing_time
                }
                
            else:
                print(f"[Error] {test_name} failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"  Response: {response.text[:200]}...")
                
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                    "processing_time": processing_time
                }
                
        except Exception as e:
            print(f"[Error] Error testing {test_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def test_unified_endpoint(self, image_path: str, test_name: str) -> Dict:
        """Test unified detection endpoint"""
        print(f"[Unified] Testing unified detection - {test_name}...")
        
        try:
            if not os.path.exists(image_path):
                print(f"[Error] Test image not found: {image_path}")
                return {"success": False, "error": "Image not found"}
            
            image_b64 = self.encode_image(image_path)
            if not image_b64:
                return {"success": False, "error": "Could not encode image"}
            
            payload = {
                "image_b64": f"data:image/jpeg;base64,{image_b64}",
                "user_id": f"test_unified_{test_name}"
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/detections/unified", 
                                   json=payload, 
                                   timeout=90)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                
                print(f"[OK] Unified detection completed in {processing_time:.2f}s")
                
                # Print summary
                if 'summary' in result_data:
                    summary = result_data['summary']
                    print(f"  [People] Total people: {summary.get('total_people', 0)}")
                    print(f"  [Head] Head direction: {summary.get('head_direction', 'unknown')}")
                    print(f"  [Body] Body visible: {'Yes' if summary.get('body_visible', False) else 'No'}")
                    print(f"  [Viol] Overall violation: {'Yes' if summary.get('overall_violation', False) else 'No'}")
                
                # Print violations
                if 'violations' in result_data:
                    violations = result_data['violations']
                    if violations.get('has_violations', False):
                        print(f"  [Violations] Violations ({violations['total_violations']}):")
                        for violation_type in violations['violation_types']:
                            print(f"    - {violation_type}")
                        else:
                            print("  [No Viol] No violations detected")
                
                # Print model types used
                if 'summary' in result_data and 'models_used' in result_data['summary']:
                    models = result_data['summary']['models_used']
                    print(f"  [Models] Models used:")
                    print(f"    - Multi-person: {models.get('multi_person', 'unknown')}")
                    print(f"    - Head pose: {models.get('head_pose', 'unknown')}")
                    print(f"    - Body visibility: {models.get('body_visibility', 'unknown')}")
                
                return {
                    "success": True,
                    "data": result_data,
                    "processing_time": processing_time
                }
                
            else:
                print(f"[Error] Unified detection failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"  Response: {response.text[:200]}...")
                
                return {
                    "success": False,
                    "error": f"Status code: {response.status_code}",
                    "processing_time": processing_time
                }
                
        except Exception as e:
            print(f"[Error] Error testing unified detection: {e}")
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_tests(self, test_images_dir: str) -> Dict:
        """Run comprehensive tests using test images"""
        print("[Start] Starting Comprehensive Testing of Integrated Detection System")
        print("=" * 70)
        
        import datetime
        self.test_results["timestamp"] = datetime.datetime.now().isoformat()
        
        # Test 1: System Status
        print("\n[Status] SYSTEM STATUS TEST")
        print("-" * 30)
        status_result = self.test_system_status()
        self.test_results["system_status"] = status_result
        
        # Test 2: Individual Endpoints
        print("\n[Individual] INDIVIDUAL ENDPOINT TESTS")
        print("-" * 30)
        
        test_images = []
        if os.path.exists(test_images_dir):
            # Find test images
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(Path(test_images_dir).glob(ext))
        
        if not test_images:
            # Use existing test image if available
            if os.path.exists("backend/test.jpg"):
                test_images = ["backend/test.jpg"]
            else:
                print("[Note] No test images found. Using sample from CrowdHuman dataset...")
                crowd_sample = Path("backend/ml/crowdhuman_light_dataset/train/images").glob("*.jpg")
                try:
                    test_images = [str(next(crowd_sample))]
                except StopIteration:
                    test_images = []
        
        if test_images:
            # Test with first available image
            test_image = str(test_images[0])
            print(f"Using test image: {test_image}")
            
            # Test individual endpoints
            endpoints_to_test = [
                ("/api/detections/multi_person", "Multi-Person Detection"),
                ("/api/detections/head_pose", "Head Pose Detection"),
                ("/api/detections/body_visibility", "Body Visibility Detection")
            ]
            
            individual_results = {}
            for endpoint, name in endpoints_to_test:
                result = self.test_individual_endpoint(endpoint, test_image, name.lower())
                individual_results[name.lower()] = result
                print()  # Add spacing
            
            self.test_results["individual_tests"] = individual_results
            
            # Test 3: Unified Endpoint
            print("\n[Unified] UNIFIED DETECTION TEST")
            print("-" * 30)
            unified_result = self.test_unified_endpoint(test_image, "comprehensive")
            self.test_results["unified_tests"] = {"comprehensive": unified_result}
            
        else:
            print("[Note] No test images available. Skipping endpoint tests.")
            self.test_results["individual_tests"] = {"error": "No test images available"}
            self.test_results["unified_tests"] = {"error": "No test images available"}
        
        # Test 4: Performance Metrics
        print("\n[Performance] PERFORMANCE SUMMARY")
        print("-" * 30)
        self.calculate_performance_metrics()
        
        return self.test_results
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics from test results"""
        metrics = {
            "system_health": "unknown",
            "models_loaded": 0,
            "average_processing_time": 0,
            "success_rate": 0,
            "custom_model_active": False
        }
        
        # System health
        if self.test_results["system_status"].get("success", False):
            status_data = self.test_results["system_status"]["data"]
            metrics["system_health"] = status_data["system"]["status"]
            metrics["models_loaded"] = status_data["system"]["models_loaded"]
            
            # Check if custom model is active
            if "models" in status_data:
                mp_status = status_data["models"]["unified_detection"]["multi_person"]
                metrics["custom_model_active"] = mp_status.get("custom_model_loaded", False)
        
        # Processing times and success rates
        processing_times = []
        successful_tests = 0
        total_tests = 0
        
        # Individual tests
        if isinstance(self.test_results.get("individual_tests", {}), dict):
            for test_name, result in self.test_results["individual_tests"].items():
                if isinstance(result, dict):
                    total_tests += 1
                    if result.get("success", False):
                        successful_tests += 1
                        if "processing_time" in result:
                            processing_times.append(result["processing_time"])
        
        # Unified tests
        if isinstance(self.test_results.get("unified_tests", {}), dict):
            for test_name, result in self.test_results["unified_tests"].items():
                if isinstance(result, dict):
                    total_tests += 1
                    if result.get("success", False):
                        successful_tests += 1
                        if "processing_time" in result:
                            processing_times.append(result["processing_time"])
        
        # Calculate metrics
        if processing_times:
            metrics["average_processing_time"] = sum(processing_times) / len(processing_times)
        
        if total_tests > 0:
            metrics["success_rate"] = (successful_tests / total_tests) * 100
        
        self.test_results["performance_metrics"] = metrics
        
        # Print metrics
        print(f"[Health] System Health: {metrics['system_health']}")
        print(f"[Models] Models Loaded: {metrics['models_loaded']}")
        print(f"[Custom] Custom CrowdHuman Model: {'Active' if metrics['custom_model_active'] else 'Inactive'}")
        print(f"[Time] Average Processing Time: {metrics['average_processing_time']:.2f}s")
        print(f"[Success] Success Rate: {metrics['success_rate']:.1f}%")
    
    def save_results(self, output_file: str):
        """Save test results to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n[Saved] Test results saved to: {output_file}")
        except Exception as e:
            print(f"\n[Error] Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test integrated detection system')
    parser.add_argument('--server_url', type=str, default='http://localhost:5000',
                       help='Flask server URL (default: http://localhost:5000)')
    parser.add_argument('--test_images_dir', type=str, default='test_images',
                       help='Directory containing test images')
    parser.add_argument('--output_file', type=str, default='test_results.json',
                       help='Output file for test results')
    parser.add_argument('--start_server', action='store_true',
                       help='Try to start the Flask server before testing')
    
    args = parser.parse_args()
    
    print("[Test] Integrated Detection System Testing Suite")
    print("=" * 50)
    
    if args.start_server:
        print("[Note] Auto-starting server not implemented. Please start manually:")
        print("  python app.py")
        print()
    
    # Create tester instance
    tester = IntegratedSystemTester(args.server_url)
    
    # Run comprehensive tests
    try:
        results = tester.run_comprehensive_tests(args.test_images_dir)
        
        # Save results
        tester.save_results(args.output_file)
        
        # Final summary
        print("\n[Summary] FINAL SUMMARY")
        print("=" * 50)
        metrics = results.get("performance_metrics", {})
        
        if metrics.get("success_rate", 0) >= 80:
            print("[Good] System is functioning well!")
        elif metrics.get("success_rate", 0) >= 60:
            print("[Warning] System has some issues but is mostly functional")
        else:
            print("[Error] System has significant issues that need attention")
        
        print(f"Overall Success Rate: {metrics.get('success_rate', 0):.1f}%")
        print(f"Custom Model Status: {'[OK] Active' if metrics.get('custom_model_active', False) else '[ERR] Inactive'}")
        
        return 0 if metrics.get("success_rate", 0) >= 70 else 1
        
    except KeyboardInterrupt:
        print("\n[Note] Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[Error] Unexpected error during testing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
