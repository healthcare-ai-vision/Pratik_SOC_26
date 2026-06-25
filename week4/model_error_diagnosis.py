import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Configuration
MODEL_PATH = 'runs/classify/train/weights/best.pt'
DIRECTORY = "/media/pratik/PENDRIVE/downloads/kaggle/test_al_kaggle_zip (1)/Combined Dataset/test/Mild Impairment/"
FILENAME = "10 (12).jpg"
IMAGE_PATH = os.path.join(DIRECTORY, FILENAME)

def generate_and_show_cam(model_path, image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and set to training mode temporarily to enable gradient tracking
    model = YOLO(model_path).model.to(device)
    model.train() # Set to train mode to ensure layers track grads
    
    # Target the last convolution layer
    target_layers = [model.model[-2]] 

    # Wrapper to handle YOLO's output tuple structure
    class YOLOv11Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)[0]

    wrapped_model = YOLOv11Wrapper(model)

    # Initialize GradCAM
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    # Load and Preprocess
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0
    
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True) # Explicitly enable grad on input

    # Generate Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Scan")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM Activation")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    generate_and_show_cam(MODEL_PATH, IMAGE_PATH)