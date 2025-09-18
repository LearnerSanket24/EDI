import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_head_pose_model(input_shape=(160, 160, 3), num_classes=5):
    """
    Create a CNN model for head pose classification
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of pose classes (forward, left, right, down, up)
    
    Returns:
        Compiled Keras model
    """
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Data augmentation layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_mobilenet_model(input_shape=(160, 160, 3), num_classes=5):
    """
    Create a MobileNet-based model for head pose classification
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of pose classes
    
    Returns:
        Compiled Keras model
    """
    
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_efficientnet_model(input_shape=(160, 160, 3), num_classes=5):
    """
    Create an EfficientNet-based model for head pose classification
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of pose classes
    
    Returns:
        Compiled Keras model
    """
    
    # Load pre-trained EfficientNetB0
    base_model = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss, and metrics
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

def get_callbacks(model_name="head_pose_model"):
    """
    Get training callbacks
    
    Args:
        model_name: Name for saving checkpoints
    
    Returns:
        List of callbacks
    """
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    return callbacks

def create_data_generators(data_dir, batch_size=32, img_size=160):
    """
    Create data generators for training and validation
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        img_size: Image size for resizing
    
    Returns:
        Tuple of (train_generator, val_generator, class_names)
    """
    
    # Data augmentation for training
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        validation_split=0.0  # We already have separate train/val folders
    )
    
    # No augmentation for validation
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.0
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    
    return train_generator, val_generator, class_names

if __name__ == "__main__":
    # Test model creation
    model = create_mobilenet_model()
    model = compile_model(model)
    print("Model created successfully!")
    print(f"Model summary:")
    model.summary()
