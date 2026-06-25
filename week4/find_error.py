import os
import torch
from ultralytics import YOLO
from pathlib import Path

# Configuration
MODEL_PATH = 'runs/classify/train/weights/best.pt'
TEST_DIR = "/media/pratik/PENDRIVE/downloads/kaggle/test_al_kaggle_zip (1)/Combined Dataset/test/"

def find_misclassifications(model_path, test_dir):
    model = YOLO(model_path)
    misclassified = []
    
    # Iterate through all classes and images
    test_path = Path(test_dir)
    for class_folder in test_path.iterdir():
        if class_folder.is_dir():
            actual_class = class_folder.name
            for image_path in class_folder.glob("*.jpg"):
                # Predict
                results = model.predict(image_path, verbose=False)
                pred_idx = results[0].probs.top1
                pred_class = results[0].names[pred_idx]
                
                # Check for error
                if pred_class != actual_class:
                    misclassified.append({
                        "path": str(image_path),
                        "actual": actual_class,
                        "predicted": pred_class
                    })
    
    # Report findings
    print(f"\n--- Misclassification Report ---")
    for error in misclassified:
        print(f"FAILED: {os.path.basename(error['path'])}")
        print(f"  Expected: {error['actual']} | Predicted: {error['predicted']}")
    
    return misclassified

if __name__ == "__main__":
    errors = find_misclassifications(MODEL_PATH, TEST_DIR)
    print(f"\nTotal errors found: {len(errors)}")