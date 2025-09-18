import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from keras_head_pose_model import (
    create_mobilenet_model, 
    create_efficientnet_model, 
    create_head_pose_model,
    compile_model, 
    get_callbacks,
    create_data_generators
)

def plot_training_history(history, save_path="training_history.png"):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, val_generator, class_names):
    """Evaluate model and generate detailed metrics"""
    print("Evaluating model...")
    
    # Get predictions
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    return y_true, y_pred, predictions

def train_model(data_dir, model_type="mobilenet", epochs=50, batch_size=32, 
                img_size=160, learning_rate=0.001, output_dir="models"):
    """Train head pose detection model"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Training {model_type} model for head pose detection...")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    
    # Create data generators
    train_generator, val_generator, class_names = create_data_generators(
        data_dir, batch_size, img_size
    )
    
    print(f"Classes: {class_names}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    
    # Create model
    if model_type == "mobilenet":
        model = create_mobilenet_model((img_size, img_size, 3), len(class_names))
    elif model_type == "efficientnet":
        model = create_efficientnet_model((img_size, img_size, 3), len(class_names))
    elif model_type == "custom":
        model = create_head_pose_model((img_size, img_size, 3), len(class_names))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compile model
    model = compile_model(model, learning_rate)
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Get callbacks
    model_name = f"head_pose_{model_type}"
    callbacks = get_callbacks(model_name)
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, f"{model_name}_history.png"))
    
    # Evaluate model
    y_true, y_pred, predictions = evaluate_model(model, val_generator, class_names)
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save model info
    model_info = {
        "model_type": model_type,
        "classes": class_names,
        "num_classes": len(class_names),
        "img_size": img_size,
        "training_samples": train_generator.samples,
        "validation_samples": val_generator.samples,
        "final_accuracy": history.history['val_accuracy'][-1],
        "best_accuracy": max(history.history['val_accuracy']),
        "epochs_trained": len(history.history['loss'])
    }
    
    with open(os.path.join(output_dir, f"{model_name}_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final validation accuracy: {model_info['final_accuracy']:.4f}")
    print(f"Best validation accuracy: {model_info['best_accuracy']:.4f}")
    
    return model, history, model_info

def main():
    parser = argparse.ArgumentParser(description="Train Keras head pose detection model")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--model_type", choices=["mobilenet", "efficientnet", "custom"], 
                       default="mobilenet", help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=160, help="Image size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output_dir", default="models", help="Output directory for models")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        return
    
    # Train model
    model, history, model_info = train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
