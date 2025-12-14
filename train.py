import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shutil

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
PLANTS = ['Apple', 'Corn', 'Grape', 'Potato']
DATASET_BASE = 'data'  # Your base data folder
MODEL_SAVE_PATH = 'models/plant_disease_model.h5'
CLASS_NAMES_PATH = 'models/class_names.json'

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data_combined/train', exist_ok=True)
os.makedirs('data_combined/val', exist_ok=True)
os.makedirs('data_combined/test', exist_ok=True)

print("="*60)
print("PREPARING DATASET")
print("="*60)

# Disease mapping to standardize names
disease_mapping = {
    # Apple
    'Apple Scab': 'Apple___Apple_scab',
    'Black Rot': 'Apple___Black_rot',
    'Cedar Apple Rust': 'Apple___Cedar_apple_rust',
    # Corn
    'Cercospora Leaf Spot': 'Corn___Cercospora_leaf_spot',
    'Common Rust': 'Corn___Common_rust',
    'Northern Leaf Blight': 'Corn___Northern_Leaf_Blight',
    # Grape
    'Esca (Black Measles)': 'Grape___Esca_(Black_Measles)',
    'Leaf Blight': 'Grape___Leaf_blight',
    # Potato
    'Early Blight': 'Potato___Early_blight',
    'Late Blight': 'Potato___Late_blight',
}

def create_combined_structure():
    """Create a combined structure for all plants"""
    print("\nCreating combined dataset structure...")
    
    for split in ['Train', 'Val', 'Test']:
        split_lower = split.lower()
        
        for plant in PLANTS:
            plant_path = os.path.join(DATASET_BASE, plant, split)
            
            if not os.path.exists(plant_path):
                print(f"Warning: {plant_path} not found!")
                continue
            
            diseases = [d for d in os.listdir(plant_path) if not d.startswith('.')]
            
            for disease in diseases:
                # Create standardized class name
                if disease.lower() == 'healthy':
                    class_name = f"{plant}___healthy"
                elif disease in disease_mapping:
                    class_name = disease_mapping[disease]
                else:
                    # Fallback: use plant name + disease
                    if plant == 'Apple' and disease not in disease_mapping:
                        class_name = f"Apple___{disease.replace(' ', '_')}"
                    elif plant == 'Grape' and disease not in disease_mapping:
                        class_name = f"Grape___{disease.replace(' ', '_')}"
                    else:
                        class_name = f"{plant}___{disease.replace(' ', '_')}"
                
                # Create symlink or copy
                src = os.path.join(plant_path, disease)
                dst = os.path.join(f'data_combined/{split_lower}', class_name)
                
                if not os.path.exists(dst):
                    try:
                        # Try symlink first (faster)
                        os.symlink(os.path.abspath(src), dst)
                    except:
                        # If symlink fails, copy
                        shutil.copytree(src, dst)
                    
                print(f"Linked: {plant}/{split}/{disease} -> {class_name}")

create_combined_structure()

print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    'data_combined/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    'data_combined/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    'data_combined/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class names
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print(f"\nDataset Statistics:")
print(f"- Total classes: {num_classes}")
print(f"- Training samples: {train_generator.samples}")
print(f"- Validation samples: {validation_generator.samples}")
print(f"- Test samples: {test_generator.samples}")
print(f"\nClasses: {class_names}")

# Save class names
with open(CLASS_NAMES_PATH, 'w') as f:
    json.dump(class_names, f, indent=2)

print("\n" + "="*60)
print("BUILDING MODEL")
print("="*60)

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

# Create model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel created successfully!")
print(f"Total parameters: {model.count_params():,}")

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("\n" + "="*60)
print("PHASE 1: INITIAL TRAINING")
print("="*60)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("PHASE 2: FINE-TUNING")
print("="*60)

# Unfreeze base model
base_model.trainable = True

# Freeze all except last 20 layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

# Test evaluation
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Validation evaluation
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Plot results
all_acc = history.history['accuracy'] + history_fine.history['accuracy']
all_val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
all_loss = history.history['loss'] + history_fine.history['loss']
all_val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(all_acc, 'b-', label='Training Accuracy', linewidth=2)
plt.plot(all_val_acc, 'r-', label='Validation Accuracy', linewidth=2)
plt.axvline(x=len(history.history['accuracy']), color='g', linestyle='--', 
            label='Fine-tuning starts', alpha=0.7)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(all_loss, 'b-', label='Training Loss', linewidth=2)
plt.plot(all_val_loss, 'r-', label='Validation Loss', linewidth=2)
plt.axvline(x=len(history.history['loss']), color='g', linestyle='--', 
            label='Fine-tuning starts', alpha=0.7)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print(f"✓ Model saved: {MODEL_SAVE_PATH}")
print(f"✓ Classes saved: {CLASS_NAMES_PATH}")
print(f"✓ Training plot: models/training_history.png")
print(f"✓ Final Test Accuracy: {test_accuracy*100:.2f}%")
print("="*60)