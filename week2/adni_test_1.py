import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def inspect_and_load_nifti(file_path):
    """
    Loads a NIfTI file, prints its shape and metadata, and returns the 3D matrix.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file at: {file_path}")
        
    # Load the NIfTI image object
    img = nib.load(file_path)
    
    print(f"--- Inspecting: {os.path.basename(file_path)} ---")
    print(f"Image Shape (X, Y, Z / Slices): {img.shape}")
    print(f"Data Type: {img.get_data_dtype()}")
    
    # Extract the actual 3D data array as floating point numbers
    data = img.get_fdata()
    return data

def preprocess_volume(data, target_shape=(128, 128, 128)):
    """
    Normalizes the intensities and prepares the 3D volume for ML.
    (Optional: You can add cropping/resizing here if scans have different shapes)
    """
    # 1. Min-Max Normalization (Scales voxel values between 0 and 1)
    # Medical images often have a wide dynamic range, normalization is critical for ML
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    
    return normalized_data

def visualize_middle_slices(data):
    """
    Plots the middle slice along all three dimensions (Sagittal, Coronal, Axial).
    """
    x_mid, y_mid, z_mid = [dim // 2 for dim in data.shape]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(data[x_mid, :, :], cmap='gray')
    axes[0].set_title('Sagittal Slice')
    
    axes[1].imshow(data[:, y_mid, :], cmap='gray')
    axes[1].set_title('Coronal Slice')
    
    axes[2].imshow(data[:, :, z_mid], cmap='gray')
    axes[2].set_title('Axial Slice')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# EXAMPLE ML PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    # Replace this with the actual path to one of your ADNI .nii files
    sample_nii_path = "ADNI_002_S_1155_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070805123441149_S33917_I64755.nii" 
    
    try:
        # 1. Load and Inspect
        raw_volume = inspect_and_load_nifti(sample_nii_path)
        
        # 2. Visualize a few slices to make sure it loaded correctly
        print("\nDisplaying middle slices...")
        visualize_middle_slices(raw_volume)
        
        # 3. Preprocess for ML
        processed_volume = preprocess_volume(raw_volume)
        print(f"Processed volume min: {processed_volume.min()}, max: {processed_volume.max()}")
        
        # 4. Mocking an ML Pipeline format
        # For a 3D CNN (like PyTorch or TensorFlow), you usually need format: (Batch, Channels, X, Y, Z)
        ml_ready_input = np.expand_dims(processed_volume, axis=(0, 1)) 
        print(f"Shape for 3D CNN input: {ml_ready_input.shape}")
        
    except FileNotFoundError as e:
        print(e)
        print("\n[Tip] Update 'sample_nii_path' with a real path to test the code.")