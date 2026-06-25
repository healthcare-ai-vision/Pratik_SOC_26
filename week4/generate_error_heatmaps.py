import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path

MODEL_PATH = 'runs/classify/train/weights/best.pt'
TEST_DIR = "/media/pratik/PENDRIVE/downloads/kaggle/test_al_kaggle_zip (1)/Combined Dataset/test/"
OUTPUT_DIR = "error_diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class YOLOv11Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)[0]

def save_error_heatmap(model, img_path, actual, predicted):
    for param in model.parameters(): param.requires_grad_(True)
    
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None: return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(next(model.parameters()).device)
    input_tensor.requires_grad_(True)

    # Use the target layer
    cam = GradCAM(model=model, target_layers=[model.model.model[-2]])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    
    # --- ADDED PLOTTING AND SAVING LOGIC ---
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(img_rgb); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(visualization); plt.title(f"Act: {actual} | Pred: {predicted}"); plt.axis("off")
    
    save_filename = f"{actual}_as_{predicted}_{img_path.name}"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path)
    plt.close() # Important: clears memory and ensures file is written

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv11Wrapper(YOLO(MODEL_PATH).model.to(device).eval())
    
    test_path = Path(TEST_DIR)
    for class_folder in test_path.iterdir():
        if class_folder.is_dir():
            actual = class_folder.name
            for img_path in class_folder.glob("*.jpg"):
                res = YOLO(MODEL_PATH).predict(img_path, verbose=False)[0]
                pred = res.names[res.probs.top1]
                if pred != actual:
                    print(f"Generating heatmap for error: {img_path.name}")
                    save_error_heatmap(model, img_path, actual, pred)
    print(f"Done! Check the folder: {os.path.abspath(OUTPUT_DIR)}")