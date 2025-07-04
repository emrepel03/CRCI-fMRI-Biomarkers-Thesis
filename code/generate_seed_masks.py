"""
Generates spherical ROI seed masks for FC analysis based on MNI coordinates.
Author: Emre Pelzer
"""

import os
import numpy as np
from nilearn import datasets
from nilearn.image import new_img_like, coord_transform
from nibabel import Nifti1Image

# Seed coordinates
seeds = {
    "mPFC": [1, 55, -3],
    "PCC": [1, -61, 38],
    "LP_left": [-39, -77, 33],
    "LP_right": [47, -67, 29],
    "ACC": [0, 22, 35],
    "AINS_left": [-44, 13, 1],
    "AINS_right": [47, 14, 0],
    "RPFC_left": [-32, 45, 27],
    "RPFC_right": [32, 46, 27],
    "SMG_left": [-60, -39, 31],
    "SMG_right": [62, -35, 32],
    "Hippocampus_left": [-28, -6, -12],
    "Hippocampus_right": [30, -8, -14],
}

radius = 6  # in mm
mni_img = datasets.load_mni152_template()
output_dir = "seed_masks"
os.makedirs(output_dir, exist_ok=True)

for name, coord in seeds.items():
    # Create a binary 3D mask with a sphere at the given coordinate
    affine = mni_img.affine
    shape = mni_img.shape
    mask_data = np.zeros(shape, dtype=np.uint8)

    # Convert MNI coordinate to voxel index
    i, j, k = np.round(coord_transform(coord[0], coord[1], coord[2], np.linalg.inv(affine))).astype(int)

    # Fill in a small sphere (radius in voxels is roughly (radius / 2) as voxel size is ~2mm)
    rr, cc, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (rr - i)**2 + (cc - j)**2 + (zz - k)**2 <= (radius // 2)**2
    mask_data[mask] = 1

    mask_img = new_img_like(mni_img, mask_data)
    path = os.path.join(output_dir, f"{name}_mask.nii.gz")
    mask_img.to_filename(path)
    print(f"✅ Saved: {path}")