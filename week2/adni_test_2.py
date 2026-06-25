import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    ScaleIntensity,
    SpatialPad,  # 🧠 Fixed: Changed from SpatialPadd to SpatialPad
    EnsureType
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm

# ==========================================
# 1. PREPROCESS YOUR SINGLE RAW .NII FILE
# ==========================================
# Pads the brain volume to a uniform size divisible by 16/32
TARGET_SHAPE = (256, 256, 256) 

preprocess = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        ScaleIntensity(),
        SpatialPad(spatial_size=TARGET_SHAPE, mode="minimum"), # Works perfectly for non-dictionary inputs
        EnsureType(),
    ]
)

# Target your local ADNI file
input_nii_path = "ADNI_002_S_1155_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070805123441149_S33917_I64755.nii"

print("Loading, padding, and preprocessing the 3D volume...")
input_tensor = preprocess(input_nii_path) 

# Format to 5D batch tensor: (Batch=1, Channel=1, X, Y, Z)
input_batch = input_tensor.unsqueeze(0)

# ==========================================
# 2. LOAD THE 3D U-NET MODEL
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2, # Background vs Brain Tissue
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# Note: Weights are randomly initialized here. 
# Load actual weights using: model.load_state_dict(torch.load("your_weights.pt"))
model.eval()

# ==========================================
# 3. RUN INFERENCE (TESTING THE DATA)
# ==========================================
print("Running the 3D volume through the network...")
with torch.no_grad():
    outputs = model(input_batch)
    
    # Extract the 3D mask matrix from the batch output
    seg_mask = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]

# ==========================================
# 4. VISUALIZE THE PREDICTED MASK
# ==========================================
raw_data = input_tensor.squeeze().cpu().numpy()
mid_slice = raw_data.shape[2] // 2 

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(raw_data[:, :, mid_slice], cmap="gray")
plt.title("Original MRI Slice")

plt.subplot(1, 2, 2)
plt.imshow(seg_mask[:, :, mid_slice], cmap="jet")
plt.title("Predicted Segmentation Mask")
plt.show()

# ==========================================
# 5. SAVE THE NEW MASK AS A .NII FILE
# ==========================================
original_nib = nib.load(input_nii_path)
new_mask_img = nib.Nifti1Image(seg_mask.astype(np.uint8), original_nib.affine)

output_path = "predicted_brain_mask.nii.gz"
nib.save(new_mask_img, output_path)
print(f"Successfully saved predicted mask to: {output_path}")