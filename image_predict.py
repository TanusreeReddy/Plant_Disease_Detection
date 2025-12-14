import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from predict import load_model_and_classes, preprocess_image
from disease_info import get_disease_info

# OUTPUT_DIR = "outputs"
IMG_SIZE = 224

def show_prediction(image_path, model, class_names):
    # Load original image for display
    img = Image.open(image_path).convert("RGB")
    # Preprocess for model
    img_array = preprocess_image(image_path)  # returns shape (1, H, W, 3)
    preds = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100.0
    predicted_class = class_names[top_idx]

    # Disease info (may contain healthy/diseased text)
    info = get_disease_info(predicted_class)

    # Build text block
    lines = [
        f"Prediction: {predicted_class} ({confidence:.2f}%)",
        f"Plant: {info.get('plant','N/A')}",
        f"Status: {info.get('status','N/A')}"
    ]
    if info.get('status') == 'Diseased':
        lines += [
            f"Disease: {info.get('disease','N/A')}",
            "",
            "Treatment:",
            info.get('treatment','N/A')
        ]
    else:
        lines += ["", "✓ Plant is healthy!"]

    text_block = "\n".join(lines)

    # Plot image and text
    plt.figure(figsize=(8, 10))
    ax = plt.subplot2grid((10, 1), (0, 0), rowspan=7)
    ax.imshow(img)
    ax.axis("off")

    # Text area below image
    txt_ax = plt.subplot2grid((10, 1), (7, 0), rowspan=3)
    txt_ax.axis("off")
    txt_ax.text(0, 1, text_block, fontsize=10, va="top", wrap=True, family="monospace")

    plt.tight_layout()

    # Do not save — just display
    plt.show()

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "disease_info": info
    }

def normalize_argv_path(argv):
    # Reconstruct path if user forgot to quote (handles spaces)
    if len(argv) < 2:
        return None
    candidate = argv[1]
    if os.path.exists(candidate):
        return candidate
    # join remaining parts and test
    joined = " ".join(argv[1:])
    if os.path.exists(joined):
        return joined
    return None

if __name__ == "__main__":
    image_path = normalize_argv_path(sys.argv)
    if not image_path:
        print("Usage: python image_predict.py \"path/to/image.jpg\"")
        sys.exit(1)

    model, class_names = load_model_and_classes()
    result = show_prediction(image_path, model, class_names)
    print(f"Prediction: {result['predicted_class']} ({result['confidence']:.2f}%)")