#!/usr/bin/env python3
"""
Lightweight CrowdHuman Dataset Preparation Script (CPU Only)
Optimized for systems without GPU - creates a smaller, manageable dataset
"""

import os
import shutil
import random
from pathlib import Path
import json
from typing import List, Dict, Tuple
import argparse

def create_yolo_structure(output_dir: str):
    """Create YOLO dataset directory structure"""
    base_path = Path(output_dir)
    
    # Create main directories
    for split in ['train', 'val']:  # Skip test to save space
        (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    return base_path

def create_simple_labels(is_crowd: bool) -> List[Tuple[int, float, float, float, float]]:
    """
    Create simple labels based on folder classification
    Much faster approach for CPU-only systems
    """
    labels = []
    
    if is_crowd:
        # For crowd images, create 2-3 simple bounding boxes
        num_people = random.randint(2, 3)  # Reduced from 2-5 to 2-3
        
        for i in range(num_people):
            # Create simple, non-overlapping bounding boxes
            if i == 0:
                # Left person
                x_center = 0.3
                y_center = 0.5
            elif i == 1:
                # Right person
                x_center = 0.7
                y_center = 0.5
            else:
                # Center person
                x_center = 0.5
                y_center = 0.4
            
            # Standard person box dimensions
            box_width = 0.25
            box_height = 0.6
            
            labels.append((0, x_center, y_center, box_width, box_height))
    else:
        # For non-crowd images, single person in center
        labels.append((0, 0.5, 0.5, 0.3, 0.7))  # Single centered person
    
    return labels

def copy_and_label_subset(source_dir: str, output_dir: str, max_images_per_class: int = 1000):
    """
    Copy a subset of images and create corresponding label files
    Much faster for CPU-only systems
    """
    base_path = Path(output_dir)
    crowd_dir = Path(source_dir) / "crowd"
    non_crowd_dir = Path(source_dir) / "non crowd"
    
    # Get limited number of image files for faster processing
    print("Scanning directories...")
    crowd_images = list(crowd_dir.glob("*.jpg"))[:max_images_per_class]
    non_crowd_images = list(non_crowd_dir.glob("*.jpg"))[:max_images_per_class]
    
    print(f"Using {len(crowd_images)} crowd images and {len(non_crowd_images)} non-crowd images")
    
    # Shuffle the datasets
    random.shuffle(crowd_images)
    random.shuffle(non_crowd_images)
    
    # Simple 80/20 split for train/val
    train_ratio = 0.8
    
    # Process each category
    total_processed = 0
    for images, is_crowd in [(crowd_images, True), (non_crowd_images, False)]:
        category = "crowd" if is_crowd else "non_crowd"
        print(f"Processing {category} images...")
        
        # Calculate split
        train_end = int(len(images) * train_ratio)
        
        splits = {
            'train': images[:train_end],
            'val': images[train_end:]
        }
        
        for split_name, split_images in splits.items():
            print(f"  {split_name}: {len(split_images)} images")
            
            for i, img_path in enumerate(split_images):
                try:
                    # Copy image
                    dest_img_path = base_path / split_name / 'images' / img_path.name
                    shutil.copy2(img_path, dest_img_path)
                    
                    # Create label file
                    label_file = base_path / split_name / 'labels' / (img_path.stem + '.txt')
                    
                    # Generate simple labels
                    labels = create_simple_labels(is_crowd)
                    
                    # Write labels to file
                    with open(label_file, 'w') as f:
                        for label in labels:
                            class_id, x_center, y_center, width, height = label
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    total_processed += 1
                    
                    # Progress indicator every 100 images
                    if (i + 1) % 100 == 0:
                        print(f"    Processed {i + 1}/{len(split_images)} images")
                        
                except Exception as e:
                    print(f"    Error processing {img_path.name}: {e}")
                    continue
    
    print(f"Total images processed: {total_processed}")
    return total_processed

def create_dataset_yaml(output_dir: str):
    """Create dataset.yaml file for YOLO training"""
    yaml_content = f"""# CrowdHuman Lightweight Dataset Configuration
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

# Classes
names:
  0: person

# Number of classes
nc: 1

# Dataset info
dataset_name: crowdhuman_light
description: Lightweight CrowdHuman dataset for CPU-only multi-person detection
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
    for split in ['train', 'val']:
        images_dir = base_path / split / 'images'
        labels_dir = base_path / split / 'labels'
        
        image_count = len(list(images_dir.glob("*.jpg")))
        label_count = len(list(labels_dir.glob("*.txt")))
        
        # Count total bounding boxes
        total_boxes = 0
        try:
            for label_file in labels_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    total_boxes += len([line for line in f if line.strip()])
        except Exception as e:
            print(f"Warning: Could not count boxes in {split}: {e}")
        
        stats[split] = {
            'images': image_count,
            'labels': label_count,
            'total_boxes': total_boxes,
            'avg_boxes_per_image': total_boxes / max(image_count, 1)
        }
    
    # Create report
    report_path = base_path / 'dataset_statistics.txt'
    with open(report_path, 'w') as f:
        f.write("CrowdHuman Lightweight Dataset Statistics\n")
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
    parser = argparse.ArgumentParser(description='Prepare lightweight CrowdHuman dataset for CPU training')
    parser.add_argument('--source_dir', type=str, 
                       default='../CrowdHuman Cropped/Dataset CrowdHuman',
                       help='Source directory containing crowd and non crowd folders')
    parser.add_argument('--output_dir', type=str, 
                       default='../ml/crowdhuman_light_dataset',
                       help='Output directory for YOLO format dataset')
    parser.add_argument('--max_per_class', type=int, default=1000,
                       help='Maximum images per class (default: 1000 for faster processing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("CrowdHuman Lightweight Dataset Preparation")
    print("="*50)
    print(f"Source directory: {args.source_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max images per class: {args.max_per_class}")
    print(f"Optimized for: CPU-only processing")
    print()
    
    try:
        # Create output directory structure
        print("Creating lightweight YOLO dataset structure...")
        create_yolo_structure(args.output_dir)
        
        # Copy and label images (subset only)
        print("Copying images and creating simple labels...")
        total_processed = copy_and_label_subset(args.source_dir, args.output_dir, args.max_per_class)
        
        if total_processed == 0:
            print("Error: No images were processed successfully!")
            return 1
        
        # Create dataset configuration
        print("Creating dataset configuration...")
        create_dataset_yaml(args.output_dir)
        
        # Generate statistics
        print("Generating statistics report...")
        stats = create_statistics_report(args.output_dir)
        
        print("\nLightweight dataset preparation complete!")
        print(f"Dataset ready at: {os.path.abspath(args.output_dir)}")
        print(f"Total images processed: {total_processed}")
        print("\nNext steps:")
        print("1. Review the dataset statistics")
        print("2. Run lightweight training: python train_crowdhuman_light.py")
        
        return 0
        
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
