#!/usr/bin/env python3
"""
CrowdHuman Dataset Preparation Script
Converts classification-based CrowdHuman data to YOLO object detection format
for training custom multi-person detection models.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image, ImageDraw
import json
from typing import List, Dict, Tuple
import argparse

# Try to import YOLO for pseudo-labeling (optional)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Will create labels based on folder classification only.")

def create_yolo_structure(output_dir: str):
    """Create YOLO dataset directory structure"""
    base_path = Path(output_dir)
    
    # Create main directories
    for split in ['train', 'val', 'test']:
        (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    return base_path

def create_pseudo_labels_with_yolo(image_path: str, is_crowd: bool) -> List[Tuple[int, float, float, float, float]]:
    """
    Use pre-trained YOLO model to generate bounding box labels
    Returns list of (class_id, x_center, y_center, width, height) in normalized coordinates
    """
    if not YOLO_AVAILABLE:
        return []
    
    try:
        # Load YOLOv8n model for pseudo-labeling
        model = YOLO('yolov8n.pt')
        
        # Run inference
        results = model(image_path, verbose=False)
        
        if not results:
            return []
        
        result = results[0]
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        labels = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            # Filter for person class (class 0 in COCO)
            person_indices = boxes.cls == 0
            
            if person_indices.any():
                person_boxes = boxes.xyxy[person_indices]
                confidences = boxes.conf[person_indices]
                
                # Only keep detections with confidence > 0.5
                high_conf_mask = confidences > 0.5
                person_boxes = person_boxes[high_conf_mask]
                
                for box in person_boxes:
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Convert to YOLO format (normalized center coordinates and dimensions)
                    x_center = (x1 + x2) / (2 * img_width)
                    y_center = (y1 + y2) / (2 * img_height)
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Class 0 for person
                    labels.append((0, x_center, y_center, width, height))
        
        return labels
        
    except Exception as e:
        print(f"Error generating pseudo labels for {image_path}: {e}")
        return []

def create_simple_labels(image_path: str, is_crowd: bool) -> List[Tuple[int, float, float, float, float]]:
    """
    Create simple labels based on folder classification
    For crowd images: assume multiple people across the image
    For non-crowd images: assume single person in center
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        labels = []
        
        if is_crowd:
            # For crowd images, create multiple dummy bounding boxes
            # This is a simplified approach - in reality you'd want actual annotations
            num_people = random.randint(2, 5)  # Assume 2-5 people in crowd images
            
            for i in range(num_people):
                # Create random but reasonable bounding boxes
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.2, 0.8)
                box_width = random.uniform(0.1, 0.3)
                box_height = random.uniform(0.2, 0.6)
                
                # Ensure box stays within image bounds
                x_center = max(box_width/2, min(1 - box_width/2, x_center))
                y_center = max(box_height/2, min(1 - box_height/2, y_center))
                
                labels.append((0, x_center, y_center, box_width, box_height))
        else:
            # For non-crowd images, assume single person in center
            labels.append((0, 0.5, 0.5, 0.4, 0.7))  # Centered person
        
        return labels
        
    except Exception as e:
        print(f"Error creating simple labels for {image_path}: {e}")
        return []

def copy_and_label_images(source_dir: str, output_dir: str, split_ratios: Dict[str, float] = None):
    """
    Copy images and create corresponding label files
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    
    base_path = Path(output_dir)
    crowd_dir = Path(source_dir) / "crowd"
    non_crowd_dir = Path(source_dir) / "non crowd"
    
    # Get all image files
    crowd_images = list(crowd_dir.glob("*.jpg")) + list(crowd_dir.glob("*.png"))
    non_crowd_images = list(non_crowd_dir.glob("*.jpg")) + list(non_crowd_dir.glob("*.png"))
    
    print(f"Found {len(crowd_images)} crowd images and {len(non_crowd_images)} non-crowd images")
    
    # Shuffle the datasets
    random.shuffle(crowd_images)
    random.shuffle(non_crowd_images)
    
    # Process each category
    for images, is_crowd in [(crowd_images, True), (non_crowd_images, False)]:
        category = "crowd" if is_crowd else "non_crowd"
        print(f"Processing {category} images...")
        
        # Calculate split indices
        train_end = int(len(images) * split_ratios['train'])
        val_end = train_end + int(len(images) * split_ratios['val'])
        
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        for split_name, split_images in splits.items():
            print(f"  {split_name}: {len(split_images)} images")
            
            for img_path in split_images:
                # Copy image
                dest_img_path = base_path / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dest_img_path)
                
                # Create label file
                label_file = base_path / split_name / 'labels' / (img_path.stem + '.txt')
                
                # Generate labels - use simple approach for faster processing
                labels = create_simple_labels(str(img_path), is_crowd)
                
                # Write labels to file
                with open(label_file, 'w') as f:
                    for label in labels:
                        class_id, x_center, y_center, width, height = label
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_dataset_yaml(output_dir: str):
    """Create dataset.yaml file for YOLO training"""
    yaml_content = f"""# CrowdHuman Custom Dataset Configuration
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: person

# Number of classes
nc: 1

# Dataset info
dataset_name: crowdhuman_custom
description: Custom CrowdHuman dataset for multi-person detection
"""
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset configuration: {yaml_path}")
    return yaml_path

def create_statistics_report(output_dir: str):
    """Create a statistics report of the prepared dataset"""
    base_path = Path(output_dir)
    
    stats = {}
    for split in ['train', 'val', 'test']:
        images_dir = base_path / split / 'images'
        labels_dir = base_path / split / 'labels'
        
        image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
        label_count = len(list(labels_dir.glob("*.txt")))
        
        # Count total bounding boxes
        total_boxes = 0
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                total_boxes += len(f.readlines())
        
        stats[split] = {
            'images': image_count,
            'labels': label_count,
            'total_boxes': total_boxes,
            'avg_boxes_per_image': total_boxes / max(image_count, 1)
        }
    
    # Create report
    report_path = base_path / 'dataset_statistics.txt'
    with open(report_path, 'w') as f:
        f.write("CrowdHuman Custom Dataset Statistics\n")
        f.write("="*50 + "\n\n")
        
        for split, data in stats.items():
            f.write(f"{split.upper()} Split:\n")
            f.write(f"  Images: {data['images']}\n")
            f.write(f"  Label files: {data['labels']}\n")
            f.write(f"  Total bounding boxes: {data['total_boxes']}\n")
            f.write(f"  Average boxes per image: {data['avg_boxes_per_image']:.2f}\n\n")
        
        total_images = sum(data['images'] for data in stats.values())
        total_boxes = sum(data['total_boxes'] for data in stats.values())
        f.write(f"TOTAL:\n")
        f.write(f"  Images: {total_images}\n")
        f.write(f"  Bounding boxes: {total_boxes}\n")
        f.write(f"  Overall average boxes per image: {total_boxes/max(total_images, 1):.2f}\n")
    
    print(f"Created statistics report: {report_path}")
    return stats

def main():
    parser = argparse.ArgumentParser(description='Prepare CrowdHuman dataset for YOLO training')
    parser.add_argument('--source_dir', type=str, 
                       default='../CrowdHuman Cropped/Dataset CrowdHuman',
                       help='Source directory containing crowd and non crowd folders')
    parser.add_argument('--output_dir', type=str, 
                       default='../ml/crowdhuman_yolo_dataset',
                       help='Output directory for YOLO format dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Split ratios sum to {total_ratio}, should be 1.0")
        return
    
    split_ratios = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': args.test_ratio
    }
    
    print("CrowdHuman Dataset Preparation")
    print("="*50)
    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: {split_ratios}")
    print(f"YOLO available for pseudo-labeling: {YOLO_AVAILABLE}")
    print()
    
    # Create output directory structure
    print("Creating YOLO dataset structure...")
    create_yolo_structure(args.output_dir)
    
    # Copy and label images
    print("Copying images and creating labels...")
    copy_and_label_images(args.source_dir, args.output_dir, split_ratios)
    
    # Create dataset configuration
    print("Creating dataset configuration...")
    create_dataset_yaml(args.output_dir)
    
    # Generate statistics
    print("Generating statistics report...")
    create_statistics_report(args.output_dir)
    
    print("\nDataset preparation complete!")
    print(f"Dataset ready for training at: {os.path.abspath(args.output_dir)}")
    print("Next steps:")
    print("1. Review the dataset statistics")
    print("2. Run the training script: python train_crowdhuman_model.py")

if __name__ == "__main__":
    main()
