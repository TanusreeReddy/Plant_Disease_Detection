import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import sys
from disease_info import get_disease_info
import os

# Configuration
MODEL_PATH = 'models/plant_disease_model.h5'
CLASS_NAMES_PATH = 'models/class_names.json'
IMG_SIZE = 224

def load_model_and_classes():
    """Load the trained model and class names"""
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def preprocess_image(image_path):
    """Preprocess the input image"""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

def predict_disease(image_path):
    """Predict disease from image"""
    print("Loading model...")
    model, class_names = load_model_and_classes()
    
    print("Processing image...")
    img_array = preprocess_image(image_path)
    
    print("Making prediction...")
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_class = class_names[predicted_class_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    print("\nTop 3 Predictions:")
    for i, idx in enumerate(top_3_idx, 1):
        print(f"{i}. {class_names[idx]}: {predictions[0][idx]*100:.2f}%")
    
    # Parse disease information
    disease_info = get_disease_info(predicted_class)
    
    print("\n" + "="*60)
    print("DISEASE INFORMATION")
    print("="*60)
    
    print(f"\nPlant: {disease_info['plant']}")
    print(f"Status: {disease_info['status']}")
    
    if disease_info['status'] == 'Diseased':
        print(f"Disease: {disease_info['disease']}")
        print(f"\nCause:\n{disease_info['cause']}")
        print(f"\nTreatment:\n{disease_info['treatment']}")
        print(f"\nPrevention:\n{disease_info['prevention']}")
    else:
        print("\n✓ Plant is healthy!")
    
    print("\n" + "="*60)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'disease_info': disease_info
    }

if __name__ == "__main__":
    import os
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py \"test_images/potato_leaf.jpg\"")
        sys.exit(1)
    
    # If the provided first arg doesn't exist and there are extra args,
    # try to join them (handles unquoted paths with spaces)
    image_path = sys.argv[1]
    if not os.path.exists(image_path) and len(sys.argv) > 2:
        candidate = " ".join(sys.argv[1:])
        if os.path.exists(candidate):
            image_path = candidate
        else:
            print(f"File not found: {image_path}")
            print("Try quoting the full path, e.g.:")
            print('  python predict.py "data\\Apple\\Test\\Black Rot\\your file.jpg"')
            sys.exit(1)

    result = predict_disease(image_path)
