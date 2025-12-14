import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = 'models/plant_disease_model.h5'
CLASS_NAMES_PATH = 'models/class_names.json'
TEST_DATA_PATH = 'data_combined/test'
RESULTS_DIR = 'results'

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("MODEL EVALUATION AND PERFORMANCE ANALYSIS")
print("="*70)

# Load model and class names
print("\n[1/6] Loading model and configuration...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded from: {MODEL_PATH}")
    
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✓ Class names loaded: {len(class_names)} classes")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load test data
print("\n[2/6] Loading test dataset...")
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important for correct label matching
)

print(f"✓ Test samples loaded: {test_generator.samples}")
print(f"✓ Number of classes: {len(test_generator.class_indices)}")

# Get predictions
print("\n[3/6] Making predictions on test set...")
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Get confidence scores
confidence_scores = np.max(predictions, axis=1)

print(f"✓ Predictions completed")
print(f"✓ Average confidence: {np.mean(confidence_scores)*100:.2f}%")

# Calculate metrics
print("\n[4/6] Calculating performance metrics...")

# Overall accuracy
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

print("\n" + "="*70)
print("OVERALL PERFORMANCE METRICS")
print("="*70)
print(f"Test Accuracy:  {test_accuracy*100:.2f}%")
print(f"Test Loss:      {test_loss:.4f}")
print(f"Total Samples:  {len(true_classes)}")
print(f"Correct:        {np.sum(predicted_classes == true_classes)}")
print(f"Incorrect:      {np.sum(predicted_classes != true_classes)}")

# Per-class metrics
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
report = classification_report(
    true_classes, 
    predicted_classes, 
    target_names=class_names,
    digits=4
)
print(report)

# Save classification report
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Total Samples: {len(true_classes)}\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*70 + "\n")
    f.write(report)

print(f"\n✓ Classification report saved to: {RESULTS_DIR}/classification_report.txt")

# Generate confusion matrix
print("\n[5/6] Generating confusion matrix...")
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Plant Disease Detection', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"✓ Confusion matrix saved to: {RESULTS_DIR}/confusion_matrix.png")

# Normalized confusion matrix (percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_normalized, 
    annot=True, 
    fmt='.2%', 
    cmap='Greens',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Percentage'}
)
plt.title('Normalized Confusion Matrix (Percentages)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix_normalized.png', dpi=150, bbox_inches='tight')
print(f"✓ Normalized confusion matrix saved to: {RESULTS_DIR}/confusion_matrix_normalized.png")

# Per-class accuracy
print("\n[6/6] Analyzing per-class performance...")

print("\n" + "="*70)
print("PER-CLASS ACCURACY")
print("="*70)
print(f"{'Class Name':<40} {'Accuracy':<12} {'Samples'}")
print("-"*70)

class_accuracies = []
for i, class_name in enumerate(class_names):
    class_mask = true_classes == i
    class_correct = np.sum((predicted_classes == true_classes) & class_mask)
    class_total = np.sum(class_mask)
    class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
    class_accuracies.append(class_acc)
    print(f"{class_name:<40} {class_acc:>6.2f}%      {class_total:>4}")

# Plot per-class accuracy
plt.figure(figsize=(12, 8))
colors = ['#2ecc71' if acc >= 90 else '#f39c12' if acc >= 80 else '#e74c3c' for acc in class_accuracies]
bars = plt.barh(class_names, class_accuracies, color=colors)
plt.xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
plt.xlim(0, 105)

# Add percentage labels
for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    plt.text(acc + 1, i, f'{acc:.1f}%', va='center', fontweight='bold')

plt.axvline(x=90, color='green', linestyle='--', alpha=0.3, label='90% threshold')
plt.axvline(x=80, color='orange', linestyle='--', alpha=0.3, label='80% threshold')
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/per_class_accuracy.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Per-class accuracy chart saved to: {RESULTS_DIR}/per_class_accuracy.png")

# Confidence score analysis
print("\n" + "="*70)
print("CONFIDENCE SCORE ANALYSIS")
print("="*70)

correct_mask = predicted_classes == true_classes
correct_confidence = confidence_scores[correct_mask]
incorrect_confidence = confidence_scores[~correct_mask]

print(f"Correct Predictions:")
print(f"  - Average confidence: {np.mean(correct_confidence)*100:.2f}%")
print(f"  - Min confidence:     {np.min(correct_confidence)*100:.2f}%")
print(f"  - Max confidence:     {np.max(correct_confidence)*100:.2f}%")

if len(incorrect_confidence) > 0:
    print(f"\nIncorrect Predictions:")
    print(f"  - Average confidence: {np.mean(incorrect_confidence)*100:.2f}%")
    print(f"  - Min confidence:     {np.min(incorrect_confidence)*100:.2f}%")
    print(f"  - Max confidence:     {np.max(incorrect_confidence)*100:.2f}%")

# Plot confidence distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(correct_confidence * 100, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Confidence (%)', fontsize=11, fontweight='bold')
plt.ylabel('Frequency', fontsize=11, fontweight='bold')
plt.title('Confidence Distribution - Correct Predictions', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

if len(incorrect_confidence) > 0:
    plt.subplot(1, 2, 2)
    plt.hist(incorrect_confidence * 100, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Confidence (%)', fontsize=11, fontweight='bold')
    plt.ylabel('Frequency', fontsize=11, fontweight='bold')
    plt.title('Confidence Distribution - Incorrect Predictions', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confidence_distribution.png', dpi=150, bbox_inches='tight')
print(f"✓ Confidence distribution saved to: {RESULTS_DIR}/confidence_distribution.png")

# Find most confused pairs
print("\n" + "="*70)
print("MOST CONFUSED CLASS PAIRS")
print("="*70)

confused_pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i][j] > 0:
            confused_pairs.append((class_names[i], class_names[j], cm[i][j]))

confused_pairs.sort(key=lambda x: x[2], reverse=True)

print(f"{'True Class':<35} {'Predicted As':<35} {'Count'}")
print("-"*70)
for true_class, pred_class, count in confused_pairs[:10]:
    print(f"{true_class:<35} {pred_class:<35} {count:>5}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Overall Test Accuracy:     {test_accuracy*100:.2f}%")
print(f"✓ Average Per-Class Accuracy: {np.mean(class_accuracies):.2f}%")
print(f"✓ Best Performing Class:      {class_names[np.argmax(class_accuracies)]} ({max(class_accuracies):.2f}%)")
print(f"✓ Worst Performing Class:     {class_names[np.argmin(class_accuracies)]} ({min(class_accuracies):.2f}%)")
print(f"✓ Classes Above 90% Accuracy: {sum(1 for acc in class_accuracies if acc >= 90)}/{len(class_accuracies)}")
print(f"✓ Average Confidence:         {np.mean(confidence_scores)*100:.2f}%")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print(f"\nAll results saved to '{RESULTS_DIR}/' directory:")
print(f"  - classification_report.txt")
print(f"  - confusion_matrix.png")
print(f"  - confusion_matrix_normalized.png")
print(f"  - per_class_accuracy.png")
print(f"  - confidence_distribution.png")
print("="*70)