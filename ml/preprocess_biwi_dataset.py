import os
import numpy as np
import cv2
from PIL import Image
import shutil
from pathlib import Path
import math

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (yaw, pitch, roll) in degrees"""
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

def classify_head_pose(yaw, pitch, roll):
    """Classify head pose into categories based on angles"""
    # Define thresholds for classification
    yaw_threshold = 15  # degrees
    pitch_threshold = 15  # degrees
    
    if abs(yaw) > yaw_threshold:
        if yaw > 0:
            return "right"
        else:
            return "left"
    elif abs(pitch) > pitch_threshold:
        if pitch > 0:
            return "up"
        else:
            return "down"
    else:
        return "forward"

def process_biwi_dataset(input_dir, output_dir):
    """Process BIWI dataset and organize into classification folders"""
    
    # Create output directory structure
    classes = ["forward", "left", "right", "down", "up"]
    for split in ["train", "val"]:
        for class_name in classes:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Process each sequence
    sequence_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.isdigit()]
    
    total_images = 0
    class_counts = {class_name: 0 for class_name in classes}
    
    for seq_dir in sorted(sequence_dirs):
        seq_path = os.path.join(input_dir, seq_dir)
        print(f"Processing sequence {seq_dir}...")
        
        # Get all pose files
        pose_files = [f for f in os.listdir(seq_path) if f.endswith('_pose.txt')]
        
        for pose_file in pose_files:
            pose_path = os.path.join(seq_path, pose_file)
            rgb_file = pose_file.replace('_pose.txt', '_rgb.png')
            rgb_path = os.path.join(seq_path, rgb_file)
            
            # Check if both files exist
            if not os.path.exists(rgb_path):
                continue
                
            try:
                # Read pose data
                with open(pose_path, 'r') as f:
                    lines = f.readlines()
                
                # Parse rotation matrix (first 3 lines)
                R = np.array([
                    [float(x) for x in lines[0].split()],
                    [float(x) for x in lines[1].split()],
                    [float(x) for x in lines[2].split()]
                ])
                
                # Convert to Euler angles
                euler_angles = rotation_matrix_to_euler_angles(R)
                yaw, pitch, roll = euler_angles
                
                # Classify pose
                pose_class = classify_head_pose(yaw, pitch, roll)
                
                # Determine train/val split (use 80/20 split)
                if int(seq_dir) <= 19:  # Use first 19 sequences for training
                    split = "train"
                else:  # Use last 5 sequences for validation
                    split = "val"
                
                # Copy and rename image
                output_filename = f"{seq_dir}_{pose_file.replace('_pose.txt', '_rgb.png')}"
                output_path = os.path.join(output_dir, split, pose_class, output_filename)
                
                # Copy image
                shutil.copy2(rgb_path, output_path)
                
                class_counts[pose_class] += 1
                total_images += 1
                
            except Exception as e:
                print(f"Error processing {pose_file}: {e}")
                continue
    
    print(f"\nDataset processing complete!")
    print(f"Total images processed: {total_images}")
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

def main():
    input_dir = "../faces_0"  # BIWI dataset directory
    output_dir = "../data/head_pose_classification"  # Output directory
    
    print("Processing BIWI Head Pose Dataset...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    process_biwi_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main()
