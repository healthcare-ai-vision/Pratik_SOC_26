from ultralytics import YOLO

# 1. Load a pre-trained model (yolo11n-cls.pt is a small, fast classification model)
model = YOLO('yolo11n-cls.pt')

# 2. Train the model
# YOLO handles the batching and image resizing automatically
results = model.train(
    data='/media/pratik/PENDRIVE/downloads/kaggle/test_al_kaggle_zip (1)/Combined Dataset',
    epochs=50,
    imgsz=224,
    device=0 # Use 'cpu' if you don't have an NVIDIA GPU
)