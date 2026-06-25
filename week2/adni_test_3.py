import nibabel as nib
import numpy as np
from skimage import measure
import trimesh

def convert_nifti_to_3d_mesh(nii_path, output_obj_path, threshold=0.3):
    """
    Extracts a 3D isosurface mesh from a NIfTI volume and saves it as an .obj file.
    """
    print(f"Loading NIfTI file: {nii_path}")
    img = nib.load(nii_path)
    data = img.get_fdata()

    # 1. Normalize data between 0 and 1 to make thresholding predictable
    data_min, data_max = np.min(data), np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min + 1e-8)

    print("Running Marching Cubes algorithm to extract 3D surface...")
    # 2. Marching Cubes calculates the 3D surface vertices and triangular faces
    verts, faces, normals, values = measure.marching_cubes(
        volume=normalized_data, 
        level=threshold
    )
    
    # 3. Correct spacing differences using the NIfTI affine matrix 
    affine = img.affine
    verts = nib.affines.apply_affine(affine, verts)

    print(f"Generated 3D Mesh with {len(verts)} vertices and {len(faces)} faces.")

    # 4. Create a 3D Mesh object and export to OBJ format
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    print(f"Saving 3D model to: {output_obj_path}")
    mesh.export(output_obj_path)
    print("Done! You can now open this file in any 3D viewer.")

# ==========================================
# CHANGE YOUR FILE NAME HERE 👇
# ==========================================
if __name__ == "__main__":
    
    # Option A: If your file is in the SAME folder as this script, just type the exact filename:
    # input_file = "your_actual_file_name.nii" 
    
    # Option B: Use the full system path if it's stored somewhere else:
    input_file = "/home/pratik/Desktop/Pratik_SOC_26/week2/ADNI_002_S_1155_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070805123441149_S33917_I64755.nii" 
    
    # Name the output file whatever you want
    output_file = "brain_model_3d.obj"
    
    # Run the generator
    convert_nifti_to_3d_mesh(input_file, output_file, threshold=0.25) 

