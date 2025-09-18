#!/usr/bin/env python3
"""
CrowdHuman Custom Model Training Script
Train a YOLOv8/v10 model specifically for multi-person detection using the prepared CrowdHuman dataset
"""

import os
import argparse
import yaml
from pathlib import Path
import json
from datetime import datetime
import shutil
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: ultralytics and/or torch not available. Please install: pip install ultralytics torch")

def load_dataset_config(dataset_path: str):
    """Load dataset configuration from yaml file"""
    yaml_path = Path(dataset_path) / 'dataset.yaml'
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, yaml_path

def create_training_config(output_dir: str, **kwargs):
    """Create training configuration file"""
    config = {
        'model_name': kwargs.get('model_name', 'yolov8n'),
        'epochs': kwargs.get('epochs', 100),
        'batch_size': kwargs.get('batch_size', 16),
        'img_size': kwargs.get('img_size', 640),
        'learning_rate': kwargs.get('lr', 0.01),
        'warmup_epochs': kwargs.get('warmup_epochs', 3),
        'momentum': kwargs.get('momentum', 0.937),
        'weight_decay': kwargs.get('weight_decay', 0.0005),
        'dropout': kwargs.get('dropout', 0.0),
        'augment': kwargs.get('augment', True),
        'patience': kwargs.get('patience', 50),
        'device': kwargs.get('device', ''),  # auto-select
        'workers': kwargs.get('workers', 8),
        'project': output_dir,
        'name': f"crowdhuman_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'pretrained': True,
        'optimizer': 'SGD',  # or 'Adam', 'AdamW'
        'close_mosaic': kwargs.get('close_mosaic', 10),  # Close mosaic augmentation in last N epochs
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,  # Use full dataset
        'plots': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every N epochs
        'cache': False,  # True for faster training but more memory usage
        'copy_paste': 0.0,  # Copy-paste augmentation probability
        'mixup': 0.0,  # Mixup augmentation probability
        'degrees': 0.0,  # Rotation degrees
        'translate': 0.1,  # Translation fraction
        'scale': 0.5,  # Scale gain
        'shear': 0.0,  # Shear degrees
        'perspective': 0.0,  # Perspective transform probability
        'flipud': 0.0,  # Vertical flip probability
        'fliplr': 0.5,  # Horizontal flip probability
        'mosaic': 1.0,  # Mosaic augmentation probability
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,  # HSV-Saturation augmentation
        'hsv_v': 0.4,  # HSV-Value augmentation
    }
    
    # Save configuration
    config_path = Path(output_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config, config_path

def setup_model(model_name: str, nc: int = 1):
    """Setup YOLO model for training"""
    if not YOLO_AVAILABLE:
        raise ImportError("YOLO not available")
    
    # Load pre-trained model
    model = YOLO(f'{model_name}.pt')
    
    # Print model info
    print(f"Loaded {model_name} model")
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"Model size (MB): {sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1024**2:.1f}")
    
    return model

def train_model(model, dataset_yaml: str, config: dict, resume: str = None):
    """Train the YOLO model"""
    training_args = {
        'data': str(dataset_yaml),
        'epochs': config['epochs'],
        'batch': config['batch_size'],
        'imgsz': config['img_size'],
        'lr0': config['learning_rate'],
        'warmup_epochs': config['warmup_epochs'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'dropout': config['dropout'],
        'augment': config['augment'],
        'patience': config['patience'],
        'device': config['device'],
        'workers': config['workers'],
        'project': config['project'],
        'name': config['name'],
        'pretrained': config['pretrained'],
        'optimizer': config['optimizer'],
        'close_mosaic': config['close_mosaic'],
        'amp': config['amp'],
        'fraction': config['fraction'],
        'plots': config['plots'],
        'save': config['save'],
        'save_period': config['save_period'],
        'cache': config['cache'],
        'copy_paste': config['copy_paste'],
        'mixup': config['mixup'],
        'degrees': config['degrees'],
        'translate': config['translate'],
        'scale': config['scale'],
        'shear': config['shear'],
        'perspective': config['perspective'],
        'flipud': config['flipud'],
        'fliplr': config['fliplr'],
        'mosaic': config['mosaic'],
        'hsv_h': config['hsv_h'],
        'hsv_s': config['hsv_s'],
        'hsv_v': config['hsv_v'],
    }
    
    if resume:
        training_args['resume'] = resume
    
    # Start training
    print("Starting training...")
    print("Training arguments:")
    for key, value in training_args.items():
        print(f"  {key}: {value}")
    print()
    
    results = model.train(**training_args)
    
    return results

def evaluate_model(model, dataset_yaml: str, config: dict):
    """Evaluate the trained model"""
    print("Evaluating model...")
    
    # Validate on test set
    validation_args = {
        'data': str(dataset_yaml),
        'imgsz': config['img_size'],
        'batch': config['batch_size'],
        'device': config['device'],
        'workers': config['workers'],
        'plots': True,
        'split': 'test'
    }
    
    results = model.val(**validation_args)
    
    return results

def export_model(model, config: dict, export_formats: list = None):
    """Export model to different formats"""
    if export_formats is None:
        export_formats = ['onnx', 'torchscript']  # Default formats
    
    print(f"Exporting model to formats: {export_formats}")
    
    export_paths = {}
    
    for format_name in export_formats:
        try:
            if format_name.lower() == 'onnx':
                path = model.export(format='onnx', imgsz=config['img_size'])
            elif format_name.lower() == 'torchscript':
                path = model.export(format='torchscript', imgsz=config['img_size'])
            elif format_name.lower() == 'coreml':
                path = model.export(format='coreml', imgsz=config['img_size'])
            elif format_name.lower() == 'tflite':
                path = model.export(format='tflite', imgsz=config['img_size'])
            else:
                print(f"Warning: Unknown export format '{format_name}'")
                continue
            
            export_paths[format_name] = path
            print(f"  {format_name}: {path}")
            
        except Exception as e:
            print(f"  Failed to export {format_name}: {e}")
    
    return export_paths

def create_deployment_package(model_path: str, config: dict, dataset_config: dict, output_dir: str):
    """Create a deployment package with all necessary files"""
    package_dir = Path(output_dir) / 'deployment_package'
    package_dir.mkdir(exist_ok=True)
    
    # Copy model files
    model_dir = package_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # Copy best model
    if os.path.exists(model_path):
        best_model_path = model_dir / 'crowdhuman_best.pt'
        shutil.copy2(model_path, best_model_path)
        print(f"Copied best model to: {best_model_path}")
    
    # Create model info file
    model_info = {
        'model_name': 'CrowdHuman Custom Multi-Person Detection',
        'base_model': config['model_name'],
        'input_size': config['img_size'],
        'classes': dataset_config.get('names', {0: 'person'}),
        'num_classes': dataset_config.get('nc', 1),
        'training_epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'created_date': datetime.now().isoformat(),
        'description': 'Custom trained YOLO model for multi-person detection using CrowdHuman dataset'
    }
    
    with open(package_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Copy configuration files
    config_dir = package_dir / 'config'
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(config_dir / 'dataset_config.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Create usage instructions
    usage_instructions = """# CrowdHuman Custom Model Usage

## Model Files
- `models/crowdhuman_best.pt`: Best trained model weights
- `config/model_info.json`: Model information and metadata
- `config/training_config.json`: Training configuration used
- `config/dataset_config.yaml`: Dataset configuration

## Quick Start

```python
from ultralytics import YOLO

# Load the custom model
model = YOLO('models/crowdhuman_best.pt')

# Run inference
results = model('path/to/image.jpg')

# Process results
for result in results:
    # Get bounding boxes
    boxes = result.boxes
    if boxes is not None:
        # Extract coordinates, confidences, and classes
        coords = boxes.xyxy  # x1, y1, x2, y2
        confs = boxes.conf   # confidence scores
        classes = boxes.cls  # class indices (0 for person)
        
        print(f"Detected {len(boxes)} persons")
        for i, (coord, conf) in enumerate(zip(coords, confs)):
            x1, y1, x2, y2 = coord
            print(f"Person {i+1}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) confidence: {conf:.3f}")
```

## Integration with Existing Project

Replace the model loading in your detection code:

```python
# Old code:
# model = YOLO('yolov8n.pt')

# New code:
model = YOLO('path/to/crowdhuman_best.pt')
```

## Model Performance
- Trained on: CrowdHuman dataset (crowd/non-crowd classification converted to object detection)
- Input size: 640x640 pixels
- Classes: person (class 0)
- Optimized for: Multi-person detection in crowded scenarios
"""
    
    with open(package_dir / 'README.md', 'w') as f:
        f.write(usage_instructions)
    
    print(f"Created deployment package at: {package_dir}")
    return package_dir

def main():
    parser = argparse.ArgumentParser(description='Train custom CrowdHuman model')
    parser.add_argument('--dataset_dir', type=str, 
                       default='../ml/crowdhuman_yolo_dataset',
                       help='Dataset directory (should contain dataset.yaml)')
    parser.add_argument('--model', type=str, default='yolov8n',
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                               'yolov10n', 'yolov10s', 'yolov10m', 'yolov10b', 'yolov10l', 'yolov10x'],
                       help='Base YOLO model to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (0 for GPU, cpu for CPU, empty for auto)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads for data loading')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='../ml/training_results',
                       help='Output directory for training results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--export_formats', nargs='+', default=['onnx', 'torchscript'],
                       help='Export formats for the trained model')
    parser.add_argument('--no_export', action='store_true',
                       help='Skip model export step')
    parser.add_argument('--no_package', action='store_true',
                       help='Skip deployment package creation')
    
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("Error: Required packages not available. Please install:")
        print("pip install ultralytics torch")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("CrowdHuman Custom Model Training")
    print("=" * 50)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Base model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Load dataset configuration
        print("Loading dataset configuration...")
        dataset_config, dataset_yaml_path = load_dataset_config(args.dataset_dir)
        print(f"Dataset: {dataset_config.get('dataset_name', 'Unknown')}")
        print(f"Classes: {dataset_config.get('names', {})}")
        print()
        
        # Create training configuration
        print("Creating training configuration...")
        training_config, config_path = create_training_config(
            str(output_dir),
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            device=args.device,
            workers=args.workers,
            patience=args.patience
        )
        print(f"Training configuration saved: {config_path}")
        print()
        
        # Setup model
        print("Setting up model...")
        model = setup_model(args.model, dataset_config.get('nc', 1))
        print()
        
        # Train model
        print("Starting training process...")
        train_results = train_model(model, dataset_yaml_path, training_config, args.resume)
        print("Training completed!")
        print()
        
        # Get path to best model
        best_model_path = Path(training_config['project']) / training_config['name'] / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            print(f"Best model saved at: {best_model_path}")
            
            # Load best model for evaluation and export
            best_model = YOLO(str(best_model_path))
            
            # Evaluate model
            print("Evaluating model...")
            eval_results = evaluate_model(best_model, dataset_yaml_path, training_config)
            print("Evaluation completed!")
            print()
            
            # Export model
            if not args.no_export:
                print("Exporting model...")
                export_paths = export_model(best_model, training_config, args.export_formats)
                print("Export completed!")
                print()
            
            # Create deployment package
            if not args.no_package:
                print("Creating deployment package...")
                package_dir = create_deployment_package(
                    str(best_model_path), 
                    training_config, 
                    dataset_config, 
                    str(output_dir)
                )
                print("Deployment package created!")
                print()
            
            print("Training and preparation completed successfully!")
            print(f"Best model: {best_model_path}")
            if not args.no_package:
                print(f"Deployment package: {package_dir}")
            print()
            print("Next steps:")
            print("1. Test the model with sample images")
            print("2. Integrate with your Flask application")
            print("3. Update the multi_person detection module")
            
        else:
            print("Warning: Best model not found at expected location")
            print("Check the training output directory for model files")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
