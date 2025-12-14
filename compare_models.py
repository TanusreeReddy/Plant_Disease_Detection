import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, VGG16, InceptionV3, EfficientNetB0
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import json
import argparse

# Configuration (defaults; can be overridden by FAST_MODE)
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15  # default comparison epochs
DATASET_BASE = 'data_combined'
RESULTS_DIR = 'model_comparison'

# New: CLI args and FAST_MODE toggle
parser = argparse.ArgumentParser(description="Compare CNN models for plant disease detection")
parser.add_argument("--fast", action="store_true", help="Enable fast mode for quick benchmarking")
parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs for training each model")
parser.add_argument("--img", type=int, default=IMG_SIZE, help="Base image size (may be reduced per model)")
parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
parser.add_argument("--steps", type=int, default=0, help="Limit steps_per_epoch (0 = use full dataset)")
parser.add_argument("--valsteps", type=int, default=0, help="Limit validation_steps (0 = use full dataset)")
args = parser.parse_args()

FAST_MODE = args.fast or os.environ.get("FAST_MODE", "0") == "1"
EPOCHS = args.epochs
IMG_SIZE = args.img
BATCH_SIZE = args.batch
STEPS_PER_EPOCH = args.steps if args.steps > 0 else None
VAL_STEPS = args.valsteps if args.valsteps > 0 else None

if FAST_MODE:
    # Aggressive speed-ups for CPU runs
    EPOCHS = min(EPOCHS, 6)
    # If not explicitly set, use small steps to sample the dataset quickly
    if STEPS_PER_EPOCH is None: STEPS_PER_EPOCH = 150
    if VAL_STEPS is None: VAL_STEPS = 50
    print("⚡ FAST_MODE enabled: epochs =", EPOCHS, "steps/valsteps =", STEPS_PER_EPOCH, VAL_STEPS)

os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*80)
print("CNN MODEL COMPARISON FOR PLANT DISEASE DETECTION")
print("="*80)

# Prepare data generators (lighter aug in FAST_MODE)
print("\n[Step 1] Loading dataset...")

train_aug_kwargs = dict(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
if FAST_MODE:
    # lighten augmentation for speed
    train_aug_kwargs.update(dict(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.0, zoom_range=0.1))

train_datagen = ImageDataGenerator(**train_aug_kwargs)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    f'{DATASET_BASE}/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    f'{DATASET_BASE}/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    f'{DATASET_BASE}/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"✓ Dataset loaded")
print(f"  - Classes: {num_classes}")
print(f"  - Train samples: {train_generator.samples}")
print(f"  - Val samples: {validation_generator.samples}")
print(f"  - Test samples: {test_generator.samples}")

# Per-model overrides: use smaller inputs for heavier nets in FAST_MODE
per_model_overrides = {
    'ResNet50': {'img_size': 192 if FAST_MODE else IMG_SIZE},
    'VGG16': {'img_size': 192 if FAST_MODE else IMG_SIZE},
    'InceptionV3': {'img_size': 192 if FAST_MODE else IMG_SIZE},
    'EfficientNetB0': {'img_size': 224 if FAST_MODE else IMG_SIZE},  # usually fine at 224
    'MobileNetV2': {'img_size': 192 if FAST_MODE else IMG_SIZE}
}

# Define models to compare
def create_model(base_model_func, model_name, img_size):
    """Create a model with given base architecture and input size"""
    print(f"\n  Building {model_name} (input {img_size}x{img_size})...")
    base_model = base_model_func(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ], name=model_name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Models to compare
models_to_test = {
    'MobileNetV2': MobileNetV2,
    'ResNet50': ResNet50,
    'VGG16': VGG16,
    'InceptionV3': InceptionV3,
    'EfficientNetB0': EfficientNetB0
}

# Store results
results = []
all_histories = {}

print("\n" + "="*80)
print("[Step 2] Training and Evaluating Models")
print("="*80)

for model_name, base_model_func in models_to_test.items():
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    try:
        # Adjust generators per model (target size)
        this_img = per_model_overrides.get(model_name, {}).get('img_size', IMG_SIZE)

        # Recreate per-model generators with the adjusted target size (shares datagen)
        train_generator_model = train_datagen.flow_from_directory(
            f'{DATASET_BASE}/train',
            target_size=(this_img, this_img),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        validation_generator_model = val_datagen.flow_from_directory(
            f'{DATASET_BASE}/val',
            target_size=(this_img, this_img),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        test_generator_model = test_datagen.flow_from_directory(
            f'{DATASET_BASE}/test',
            target_size=(this_img, this_img),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        # Create model
        model = create_model(base_model_func, model_name, this_img)

        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3 if FAST_MODE else 5, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2 if FAST_MODE else 3, min_lr=1e-7, verbose=1)
        ]

        # Train model (limit steps for speed if requested)
        print(f"  Training {model_name}...")
        start_time = time.time()

        history = model.fit(
            train_generator_model,
            validation_data=validation_generator_model,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VAL_STEPS
        )

        training_time = time.time() - start_time

        # Evaluate on test set
        print(f"  Evaluating {model_name}...")
        test_loss, test_accuracy = model.evaluate(test_generator_model, verbose=0)

        # Get best validation accuracy
        best_val_accuracy = max(history.history.get('val_accuracy', [0]))
        best_train_accuracy = max(history.history.get('accuracy', [0]))

        # Store results
        results.append({
            'Model': model_name,
            'Train Accuracy': f"{best_train_accuracy*100:.2f}%",
            'Val Accuracy': f"{best_val_accuracy*100:.2f}%",
            'Test Accuracy': f"{test_accuracy*100:.2f}%",
            'Test Loss': f"{test_loss:.4f}",
            'Parameters': f"{total_params:,}",
            'Training Time (sec)': f"{training_time:.1f}",
            'Epochs Trained': len(history.history.get('loss', []))
        })

        all_histories[model_name] = history.history

        print(f"  ✓ {model_name} completed!")
        print(f"    - Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"    - Training Time: {training_time:.1f} seconds")

        # Save model (native Keras format recommended)
        save_path = f'{RESULTS_DIR}/{model_name.lower()}_model.keras'
        try:
            model.save(save_path)
        except Exception:
            # fallback to legacy h5 if needed
            model.save(f'{RESULTS_DIR}/{model_name.lower()}_model.h5')

        # Clear memory
        del model
        keras.backend.clear_session()

    except Exception as e:
        print(f"  ✗ Error with {model_name}: {str(e)}")
        results.append({
            'Model': model_name,
            'Train Accuracy': 'Error',
            'Val Accuracy': 'Error',
            'Test Accuracy': 'Error',
            'Test Loss': 'Error',
            'Parameters': 'Error',
            'Training Time (sec)': 'Error',
            'Epochs Trained': 'Error'
        })

# Create comparison table
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))

# Save to CSV
df.to_csv(f'{RESULTS_DIR}/model_comparison_results.csv', index=False)
print(f"\n✓ Results saved to: {RESULTS_DIR}/model_comparison_results.csv")

# Plot 1: Test Accuracy Comparison
print("\n[Step 3] Generating comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Extract numeric values for plotting
model_names = [r['Model'] for r in results if r['Test Accuracy'] != 'Error']
test_accuracies = [float(r['Test Accuracy'].strip('%')) for r in results if r['Test Accuracy'] != 'Error']
train_accuracies = [float(r['Train Accuracy'].strip('%')) for r in results if r['Train Accuracy'] != 'Error']
val_accuracies = [float(r['Val Accuracy'].strip('%')) for r in results if r['Val Accuracy'] != 'Error']
training_times = [float(r['Training Time (sec)']) for r in results if r['Training Time (sec)'] != 'Error']

# Plot 1: Test Accuracy
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
bars1 = axes[0, 0].bar(model_names, test_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Test Accuracy Comparison', fontweight='bold', fontsize=13, pad=15)
axes[0, 0].set_ylim(0, 100)
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, acc in zip(bars1, test_accuracies):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: Train vs Val vs Test
x = np.arange(len(model_names))
width = 0.25

bars1 = axes[0, 1].bar(x - width, train_accuracies, width, label='Train', color='#3498db', alpha=0.8)
bars2 = axes[0, 1].bar(x, val_accuracies, width, label='Validation', color='#2ecc71', alpha=0.8)
bars3 = axes[0, 1].bar(x + width, test_accuracies, width, label='Test', color='#e74c3c', alpha=0.8)

axes[0, 1].set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11)
axes[0, 1].set_title('Train vs Validation vs Test Accuracy', fontweight='bold', fontsize=13, pad=15)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
axes[0, 1].legend(fontsize=10)
axes[0, 1].set_ylim(0, 105)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Training Time
bars3 = axes[1, 0].bar(model_names, training_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[1, 0].set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Training Time Comparison', fontweight='bold', fontsize=13, pad=15)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

for bar, time_val in zip(bars3, training_times):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.02,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 4: Training History (Validation Accuracy)
for model_name, history in all_histories.items():
    if 'val_accuracy' in history:
        axes[1, 1].plot(history['val_accuracy'], label=model_name, linewidth=2, marker='o', markersize=4)

axes[1, 1].set_xlabel('Epoch', fontweight='bold', fontsize=11)
axes[1, 1].set_ylabel('Validation Accuracy', fontweight='bold', fontsize=11)
axes[1, 1].set_title('Validation Accuracy Over Epochs', fontweight='bold', fontsize=13, pad=15)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/model_comparison_charts.png', dpi=150, bbox_inches='tight')
print(f"✓ Comparison charts saved to: {RESULTS_DIR}/model_comparison_charts.png")

# Summary recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if test_accuracies:
    best_accuracy_idx = test_accuracies.index(max(test_accuracies))
    best_model = model_names[best_accuracy_idx]
    best_accuracy = test_accuracies[best_accuracy_idx]
    
    fastest_idx = training_times.index(min(training_times))
    fastest_model = model_names[fastest_idx]
    fastest_time = training_times[fastest_idx]
    
    print(f"\n✓ Best Accuracy: {best_model} ({best_accuracy:.2f}%)")
    print(f"✓ Fastest Training: {fastest_model} ({fastest_time:.1f} seconds)")
    
    # Calculate accuracy/time ratio
    efficiency_scores = [acc/time for acc, time in zip(test_accuracies, training_times)]
    best_efficiency_idx = efficiency_scores.index(max(efficiency_scores))
    best_efficiency_model = model_names[best_efficiency_idx]
    
    print(f"✓ Best Efficiency (Accuracy/Time): {best_efficiency_model}")
    
    print("\n" + "-"*80)
    print("Model Selection Guide:")
    print("-"*80)
    print(f"• For HIGHEST ACCURACY: Use {best_model}")
    print(f"• For FASTEST TRAINING: Use {fastest_model}")
    print(f"• For BEST BALANCE: Use {best_efficiency_model}")
    print(f"• For DEPLOYMENT (mobile/edge): Use MobileNetV2 or EfficientNetB0")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
print(f"\nAll results saved in '{RESULTS_DIR}/' directory:")
print(f"  • model_comparison_results.csv")
print(f"  • model_comparison_charts.png")
print(f"  • Individual model files (.h5)")
print("="*80)