import torch
import glob
import os
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Resize # 🧠 Changed SpatialPad to Resize
from monai.data import DataLoader, Dataset
from monai.networks.nets import AutoEncoder

# 1. Load all your raw ADNI files
file_paths = sorted(glob.glob("/home/pratik/Desktop/Pratik_SOC_26/week2/*.nii*"))

if not file_paths:
    raise FileNotFoundError("No .nii or .nii.gz files found in the specified directory!")

print(f"Found {len(file_paths)} NIfTI files for training.")

# 2. Preprocessing Pipeline
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize(spatial_size=(128, 128, 128), mode="trilinear"), # 👈 Forces a perfect 3D cube that downsamples cleanly
    EnsureType()
])

dataset = Dataset(data=file_paths, transform=transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 3. Define 3D Autoencoder Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64),
    strides=(2, 2, 2)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = torch.nn.MSELoss() 

# 4. Training Loop
print("Starting Unsupervised Training on Raw NIfTI files...")
for epoch in range(10):
    model.train()
    total_loss = 0
    step = 0
    
    for batch in loader:
        step += 1
        inputs = batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Self-supervised: Both tensors are now guaranteed to be (1, 1, 128, 128, 128)
        loss = loss_function(outputs, inputs) 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Epoch [{epoch+1}/10], Step [{step}/{len(loader)}], Loss: {loss.item():.4f}")
        
    print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(loader):.4f} ---")

print("Training finished successfully!")