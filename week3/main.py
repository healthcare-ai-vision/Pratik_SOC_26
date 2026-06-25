import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_visualize_nifti(file_path):
    if not os.path.exists(file_path):
        print(f" Error: File not found at {file_path}")
        return

    # =====================================================================
    # 1. NIfTI METADATA ANALYSIS
    # =====================================================================
    # Load the NIfTI object cleanly with no trailing spaces
    nii_obj = nib.load(file_path)
    
    # Extract raw structural data matrix
    data_matrix = nii_obj.get_fdata()
    
    # Extract header information
    header = nii_obj.header

    print("NIfTI FILE ANALYSIS ")
    print(f" File Path:       {os.path.basename(file_path)}")
    print(f" Data Shape:       {data_matrix.shape} (X, Y, Z coordinates)")
    print(f" Total Voxels:     {data_matrix.size:,}")
    print(f" Voxel Size (mm):  {header.get_zooms()[:3]} (Spacing between pixels)")
    print(f" Time Dimension:   {header.get_zooms()[3] if len(header.get_zooms()) > 3 else '3D Volume Only'}")
    print(f" Data Type (Raw):  {header.get_data_dtype()}")
    print(f" Intensity Range:  Min = {np.min(data_matrix):.4f} | Max = {np.max(data_matrix):.4f}")
    
    # Spatial Coordinate Orientation (e.g., RAS, LPI)
    orientation = nib.aff2axcodes(nii_obj.affine)
    print(f" Brain Orientation:{orientation}")
    print("=============================================================\n")

    # =====================================================================
    # 2. ORTHOGONAL VISUALIZATION VIA MATPLOTLIB
    # =====================================================================
    # Compute center slices for X, Y, and Z planes
    mid_x = data_matrix.shape[0] // 2
    mid_y = data_matrix.shape[1] // 2
    mid_z = data_matrix.shape[2] // 2

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Orthogonal View: {os.path.basename(file_path)}", fontsize=14, fontweight='bold')

    # 1. Axial Slice (Z-Plane cut, looking down from top)
    plt.subplot(1, 3, 1)
    axial_slice = data_matrix[:, :, mid_z]
    plt.imshow(np.rot90(axial_slice), cmap='gray')
    plt.title(f"Axial Slice (Index {mid_z})")
    plt.axis('off')

    # 2. Coronal Slice (Y-Plane cut, looking from the front)
    plt.subplot(1, 3, 2)
    coronal_slice = data_matrix[:, mid_y, :]
    plt.imshow(np.rot90(coronal_slice), cmap='gray')
    plt.title(f"Coronal Slice (Index {mid_y})")
    plt.axis('off')

    # 3. Sagittal Slice (X-Plane cut, looking from the side)
    plt.subplot(1, 3, 3)
    sagittal_slice = data_matrix[mid_x, :, :]
    plt.imshow(np.rot90(sagittal_slice), cmap='gray')
    plt.title(f"Sagittal Slice (Index {mid_x})")
    plt.axis('off')

    plt.tight_layout()
    print(" Rendering 3-plane visualization dashboard...")
    plt.show()

# =====================================================================
# RUN APPLICATION
# =====================================================================
if __name__ == '__main__':
    # Using your actual local file verified by 'ls'
    target_scan = "ADNI_005_S_10835_20260514192327060.nii"
    
    analyze_and_visualize_nifti(target_scan)
