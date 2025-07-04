"""
Performs seed-based FC extraction by computing correlation maps from fMRI and seed masks.
Author: Emre Pelzer
"""

import json
import os
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiSpheresMasker
from nilearn.image import coord_transform

# List of seed regions with MNI coordinates from Krönke et al. (2020)
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

output_path = os.path.join("datasets", "ds004796", "seed_coordinates.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(seeds, f, indent=4)

print(f"Saved seed coordinates to: {output_path}")

# Path to preprocessed data and output
preproc_dir = "/Users/emrepelzer/Desktop/THESIS/datasets/ds004796/preprocessed"
output_dir = "/Users/emrepelzer/Desktop/THESIS/datasets/ds004796/features_fc"
os.makedirs(output_dir, exist_ok=True)


# List all subject files
subject_files = [f for f in os.listdir(preproc_dir) if f.endswith("_mc.nii.gz")]

from nilearn.image import coord_transform

for file in subject_files:
    subject_id = file.split("_")[0]
    img_path = os.path.join(preproc_dir, file)
    img = nib.load(img_path)
    affine = img.affine

    try:
        full_data = img.get_fdata()
        n_timepoints = full_data.shape[-1]

        for seed_name, mni_coord in seed_coords.items():
            voxel_coord = coord_transform(*mni_coord, np.linalg.inv(affine))
            voxel_coord_rounded = tuple(np.round(voxel_coord).astype(int))

            x, y, z = voxel_coord_rounded
            if not (0 <= x < full_data.shape[0] and 0 <= y < full_data.shape[1] and 0 <= z < full_data.shape[2]):
                print(f"⚠️ Skipped {subject_id} - {seed_name}: voxel out of bounds")
                continue

            seed_ts = full_data[x, y, z, :]
            if np.std(seed_ts) == 0:
                print(f"⚠️ Skipped {subject_id} - {seed_name}: zero std in seed time series")
                continue

            all_vox_ts = full_data.reshape(-1, n_timepoints)
            correlations = np.array([
                np.corrcoef(seed_ts, voxel_ts)[0, 1] if np.std(voxel_ts) > 0 else 0
                for voxel_ts in all_vox_ts
            ])
            fc_map = correlations.reshape(full_data.shape[:3])

            output_path = os.path.join(output_dir, f"{subject_id}_{seed_name}_fc.npy")
            np.save(output_path, fc_map)

    except Exception as e:
        print(f"❌ Error processing {subject_id}: {e}")
