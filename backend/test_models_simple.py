#!/usr/bin/env python3
"""
Simple test script to verify both multi-person detection and head pose models work correctly
"""

import os
from pathlib import Path
from detections.multi_person import infer_multi_person
from detections.head_pose import infer_head_pose

def test_with_image(image_path: str):
    """Test both models with a single image"""
    print(f"Testing with image: {image_path}")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Read image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        print(f"📸 Image size: {len(image_bytes)} bytes")
        
        # Test multi-person detection
        print("\n🔍 Testing Multi-Person Detection...")
        mp_result = infer_multi_person(image_bytes)
        
        print(f"  👥 People detected: {mp_result.get('num_people', 0)}")
        print(f"  🎯 Confidence: {mp_result.get('confidence', 0):.3f}")
        print(f"  ⚠️ Violation: {'Yes' if mp_result.get('violation', False) else 'No'}")
        print(f"  🤖 Model: {mp_result.get('model_type', 'Unknown')}")
        
        if 'debug_info' in mp_result:
            debug = mp_result['debug_info']
            print(f"  🔧 Debug - Threshold: {debug.get('confidence_threshold', 0)}")
            print(f"  🔧 Debug - Raw detections: {debug.get('raw_detections', 0)}")
            print(f"  🔧 Debug - Filtered: {debug.get('filtered_detections', 0)}")
        
        if mp_result.get('error'):
            print(f"  ❌ Error: {mp_result['error']}")
        
        # Test head pose detection
        print("\n🔍 Testing Head Pose Detection...")
        hp_result = infer_head_pose(image_bytes)
        
        if 'direction' in hp_result:
            print(f"  👤 Head direction: {hp_result.get('direction', 'unknown')}")
            print(f"  🎯 Confidence: {hp_result.get('confidence', 0):.3f}")
            print(f"  ⚠️ Violation: {'Yes' if hp_result.get('violation', False) else 'No'}")
            print(f"  🤖 Using trained model: {'Yes' if hp_result.get('using_trained', False) else 'No'}")
        elif 'pose' in hp_result:
            print(f"  👤 Head pose: {hp_result.get('pose', 'unknown')}")
            print(f"  🎯 Confidence: {hp_result.get('confidence', 0):.3f}")
            print(f"  ⚠️ Violation: {'Yes' if hp_result.get('violation', False) else 'No'}")
            print(f"  🤖 Using trained model: {'Yes' if hp_result.get('using_trained', False) else 'No'}")
        
        if hp_result.get('error'):
            print(f"  ❌ Error: {hp_result['error']}")
        
        # Summary
        print("\n📊 Summary:")
        mp_ok = mp_result.get('num_people', 0) >= 0 and not mp_result.get('error')
        hp_ok = ('direction' in hp_result or 'pose' in hp_result) and not hp_result.get('error')
        
        print(f"  Multi-person detection: {'✅ Working' if mp_ok else '❌ Issues'}")
        print(f"  Head pose detection: {'✅ Working' if hp_ok else '❌ Issues'}")
        
        return mp_ok and hp_ok
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def main():
    print("🧪 Simple Model Testing")
    print("=" * 50)
    
    # Try to find a test image
    test_images = []
    
    # Check for existing test image
    if os.path.exists("test.jpg"):
        test_images.append("test.jpg")
    
    # Check CrowdHuman dataset
    crowd_dir = Path("../CrowdHuman Cropped/Dataset CrowdHuman/crowd")
    if crowd_dir.exists():
        crowd_images = list(crowd_dir.glob("*.jpg"))
        if crowd_images:
            test_images.append(str(crowd_images[0]))
    
    non_crowd_dir = Path("../CrowdHuman Cropped/Dataset CrowdHuman/non crowd")
    if non_crowd_dir.exists():
        non_crowd_images = list(non_crowd_dir.glob("*.jpg"))
        if non_crowd_images:
            test_images.append(str(non_crowd_images[0]))
    
    if not test_images:
        print("❌ No test images found!")
        print("Please ensure you have:")
        print("  - test.jpg in the current directory, or")
        print("  - CrowdHuman dataset in ../CrowdHuman Cropped/Dataset CrowdHuman/")
        return 1
    
    # Test with available images
    success_count = 0
    for i, image_path in enumerate(test_images[:2]):  # Test max 2 images
        print(f"\n🔄 Test {i+1}/{min(len(test_images), 2)}")
        success = test_with_image(image_path)
        if success:
            success_count += 1
        print()
    
    # Final result
    print("🎯 FINAL RESULTS")
    print("=" * 50)
    total_tests = min(len(test_images), 2)
    success_rate = (success_count / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 100:
        print("🎉 All models are working correctly!")
        return 0
    elif success_rate >= 50:
        print("⚠️ Some models have issues, but basic functionality works")
        return 0
    else:
        print("❌ Major issues detected. Please check model setup.")
        return 1

if __name__ == "__main__":
    exit(main())
