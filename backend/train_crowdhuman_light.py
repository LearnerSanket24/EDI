#!/usr/bin/env python3
"""
Lightweight CrowdHuman Model Training Script (CPU Only)
Optimized for CPU training with minimal resource usage
"""

import os
import argparse
import yaml
from pathlib import Path
import json
from datetime import datetime
import shutil

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: ultralytics not available. Please install: pip install ultralytics")

def load_dataset_config(dataset_path: str):
    """Load dataset configuration from yaml file"""
    yaml_path = Path(dataset_path) / 'dataset.yaml'
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, yaml_path

def create_lightweight_training_config(output_dir: str, **kwargs):
    """Create lightweight training configuration optimized for CPU"""
    config = {
        'model_name': kwargs.get('model_name', 'yolov8n'),
        'epochs': kwargs.get('epochs', 50),  # Reduced from 100
        'batch_size': kwargs.get('batch_size', 4),  # Much smaller batch size for CPU
        'img_size': kwargs.get('img_size', 320),  # Smaller image size for CPU
        'learning_rate': kwargs.get('lr', 0.001),  # Lower learning rate
        'warmup_epochs': kwargs.get('warmup_epochs', 1),  # Reduced warmup
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'dropout': 0.0,
        'augment': False,  # Disable augmentation for faster training
        'patience': kwargs.get('patience', 20),  # Reduced patience
        'device': 'cpu',  # Force CPU
        'workers': kwargs.get('workers', 2),  # Fewer workers
        'project': output_dir,
        'name': f"crowdhuman_light_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'pretrained': True,
        'optimizer': 'Adam',  # Adam often works better with smaller batches
        'close_mosaic': 10,
        'amp': False,  # Disable AMP for CPU
        'fraction': 1.0,
        'plots': False,  # Disable plots to save time
        'save': True,
        'save_period': 20,  # Save less frequently
        'cache': False,
        'copy_paste': 0.0,
        'mixup': 0.0,
        'degrees': 0.0,
        'translate': 0.0,  # Disable translation
        'scale': 0.0,  # Disable scaling
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,  # Disable flipping for faster training
        'mosaic': 0.0,  # Disable mosaic
        'hsv_h': 0.0,  # Disable HSV augmentation
        'hsv_s': 0.0,
        'hsv_v': 0.0,
    }
    
    # Save configuration
    config_path = Path(output_dir) / 'lightweight_training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config, config_path

def setup_lightweight_model(model_name: str):
    """Setup YOLO model for lightweight training"""
    if not YOLO_AVAILABLE:
        raise ImportError("YOLO not available")
    
    # Force YOLOv8n for CPU training (smallest model)
    if not model_name.endswith('n'):
        print(f"Switching from {model_name} to yolov8n for CPU training")
        model_name = 'yolov8n'
    
    # Load pre-trained model
    model = YOLO(f'{model_name}.pt')
    
    print(f"Loaded {model_name} model for CPU training")
    print("Model optimized for CPU-only processing")
    
    return model

def train_lightweight_model(model, dataset_yaml: str, config: dict):
    """Train the YOLO model with lightweight settings"""
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
        'verbose': False,  # Reduce output
    }
    
    print("Starting lightweight training...")
    print("Training configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Image size: {config['img_size']}")
    print(f"  Device: {config['device']}")
    print(f"  Workers: {config['workers']}")
    print()
    
    results = model.train(**training_args)
    
    return results

def create_integration_files(model_path: str, config: dict, output_dir: str):
    """Create files needed for integration with the Flask app"""
    integration_dir = Path(output_dir) / 'integration'
    integration_dir.mkdir(exist_ok=True)
    
    # Copy model to backend directory
    backend_models_dir = Path("../backend/models")
    backend_models_dir.mkdir(exist_ok=True)
    
    custom_model_path = backend_models_dir / "crowdhuman_custom.pt"
    if os.path.exists(model_path):
        shutil.copy2(model_path, custom_model_path)
        print(f"Copied trained model to: {custom_model_path}")
    
    # Create updated multi-person detection module
    updated_detection_code = f'''from typing import Dict
import base64
import io
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import numpy as np
except Exception:
    np = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

_custom_model = None
_fallback_model = None

def _get_custom_model():
    global _custom_model
    if _custom_model is None and YOLO is not None:
        try:
            # Try to load custom trained model first
            _custom_model = YOLO('models/crowdhuman_custom.pt')
            print("Loaded custom CrowdHuman model")
        except Exception as e:
            print(f"Could not load custom model: {{e}}")
            _custom_model = False
    return _custom_model

def _get_fallback_model():
    global _fallback_model
    if _fallback_model is None and YOLO is not None:
        # Fallback to pre-trained YOLOv8n
        _fallback_model = YOLO('yolov8n.pt')
        print("Loaded fallback YOLOv8n model")
    return _fallback_model

def _bytes_to_numpy(image_bytes: bytes):
    if Image is None:
        return None
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if np is None:
        return None
    return np.array(img)

def infer_multi_person_custom(image_bytes: bytes) -> Dict:
    """Multi-person detection using custom trained CrowdHuman model with fallback"""
    
    # Try custom model first
    model = _get_custom_model()
    using_custom = True
    
    # Fallback to pre-trained model if custom fails
    if model is False or model is None:
        model = _get_fallback_model()
        using_custom = False
        if model is None:
            return {{"num_people": 0, "confidence": 0.0, "violation": False, "using_custom": False, "error": "No models available"}}

    frame = _bytes_to_numpy(image_bytes)
    if frame is None:
        return {{"num_people": 0, "confidence": 0.0, "violation": False, "using_custom": using_custom, "error": "Could not process image"}}
    
    try:
        # Run inference with appropriate confidence threshold
        conf_threshold = 0.3 if using_custom else 0.25  # Lower threshold for custom model
        results = model.predict(source=frame, classes=[0], conf=conf_threshold, verbose=False)
        
        if not results:
            return {{"num_people": 0, "confidence": 0.0, "violation": False, "using_custom": using_custom}}

        r = results[0]
        num_people = int(len(r.boxes)) if r.boxes is not None else 0
        conf = float(r.boxes.conf.max().item()) if num_people > 0 else 0.0
        violation = num_people > 1
        
        return {{
            "num_people": num_people, 
            "confidence": conf, 
            "violation": violation, 
            "using_custom": using_custom,
            "model_type": "CrowdHuman Custom" if using_custom else "YOLOv8n Fallback"
        }}
        
    except Exception as e:
        return {{"num_people": 0, "confidence": 0.0, "violation": False, "using_custom": using_custom, "error": str(e)}}
'''
    
    # Save the updated detection module
    with open(integration_dir / 'multi_person_custom.py', 'w') as f:
        f.write(updated_detection_code)
    
    # Create integration instructions
    integration_instructions = f"""# Integration Instructions

## Files Created:
1. `{custom_model_path}` - Trained CrowdHuman model
2. `integration/multi_person_custom.py` - Updated detection module

## Integration Steps:

### Step 1: Update the detection module
Replace the contents of `detections/multi_person_yolov8.py` with the contents of `integration/multi_person_custom.py`

Or create a new file `detections/multi_person_custom.py` and update `detections/multi_person.py` to import from it.

### Step 2: Test the integration
Run your Flask app and test the `/api/detections/multi_person` endpoint with sample images.

### Step 3: Verify model loading
Check the `/api/debug/model` endpoint to see model status.

## Model Information:
- Base model: {config['model_name']}
- Training epochs: {config['epochs']}
- Image size: {config['img_size']}
- Optimized for: CPU-only inference
- Classes: person (class 0)

## Expected Performance:
- Single person images: num_people=1, violation=False
- Crowd images: num_people>1, violation=True
- Better accuracy on CrowdHuman-style images compared to generic YOLO

## Troubleshooting:
- If custom model fails to load, system automatically falls back to YOLOv8n
- Check console output for model loading messages
- Verify model file exists at: {custom_model_path}
"""
    
    with open(integration_dir / 'INTEGRATION_GUIDE.txt', 'w') as f:
        f.write(integration_instructions)
    
    print(f"Created integration files in: {integration_dir}")
    return integration_dir

def main():
    parser = argparse.ArgumentParser(description='Train lightweight CrowdHuman model for CPU')
    parser.add_argument('--dataset_dir', type=str, 
                       default='../ml/crowdhuman_light_dataset',
                       help='Dataset directory (should contain dataset.yaml)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50 for CPU)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training (default: 4 for CPU)')
    parser.add_argument('--img_size', type=int, default=320,
                       help='Image size for training (default: 320 for CPU)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of worker threads (default: 2 for CPU)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    parser.add_argument('--output_dir', type=str, default='../ml/training_results_light',
                       help='Output directory for training results')
    
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("Error: Required packages not available. Please install:")
        print("pip install ultralytics")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("CrowdHuman Lightweight Model Training (CPU Only)")
    print("=" * 60)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Workers: {args.workers}")
    print(f"Output directory: {output_dir}")
    print("Optimization: CPU-only training with minimal resources")
    print()
    
    try:
        # Load dataset configuration
        print("Loading dataset configuration...")
        dataset_config, dataset_yaml_path = load_dataset_config(args.dataset_dir)
        print(f"Dataset: {dataset_config.get('dataset_name', 'Unknown')}")
        print(f"Classes: {dataset_config.get('names', {})}")
        print()
        
        # Create training configuration
        print("Creating lightweight training configuration...")
        training_config, config_path = create_lightweight_training_config(
            str(output_dir),
            model_name='yolov8n',  # Force nano model for CPU
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            lr=args.lr,
            workers=args.workers,
            patience=args.patience
        )
        print(f"Training configuration saved: {config_path}")
        print()
        
        # Setup model
        print("Setting up lightweight model...")
        model = setup_lightweight_model('yolov8n')
        print()
        
        # Train model
        print("Starting lightweight training process...")
        print("Note: This may take 30-60 minutes depending on your CPU")
        train_results = train_lightweight_model(model, dataset_yaml_path, training_config)
        print("Training completed!")
        print()
        
        # Get path to best model
        best_model_path = Path(training_config['project']) / training_config['name'] / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            print(f"Best model saved at: {best_model_path}")
            
            # Create integration files
            print("Creating integration files...")
            integration_dir = create_integration_files(str(best_model_path), training_config, str(output_dir))
            
            print("\\nTraining and integration preparation completed!")
            print(f"Best model: {best_model_path}")
            print(f"Integration files: {integration_dir}")
            print()
            print("Next steps:")
            print("1. Review the integration guide in the output directory")
            print("2. Copy the model to your backend/models/ directory")
            print("3. Update your Flask app detection modules")
            print("4. Test the integrated solution")
            
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
