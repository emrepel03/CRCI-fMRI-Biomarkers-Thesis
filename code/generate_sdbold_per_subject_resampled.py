"""
Computes voxelwise SD-BOLD maps per subject and saves extracted features.
Author: Emre Pelzer
"""

import os
import nibabel as nib
import pandas as pd
import numpy as np
from nilearn.image import resample_to_img

input_dir = "features_sdbold"
mask_dir = "seed_masks"
output_csv = "sdbold_per_subject.csv"

records = []

# Load seed masks
seed_masks = {}
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith("_mask.nii.gz"):
        seed_name = mask_file.replace("_mask.nii.gz", "")
        mask_path = os.path.join(mask_dir, mask_file)
        seed_masks[seed_name] = nib.load(mask_path)

# Process each subject's SD-BOLD map
for fname in os.listdir(input_dir):
    if fname.endswith(".nii") or fname.endswith(".nii.gz"):
        subject_id = fname.split("_")[0]
        sdbold_path = os.path.join(input_dir, fname)
        sdbold_img = nib.load(sdbold_path)
        sdbold_data = sdbold_img.get_fdata()
        record = {"subject": subject_id}

        for seed, seed_mask_img in seed_masks.items():
            try:
                resampled_mask = resample_to_img(seed_mask_img, sdbold_img, interpolation="nearest", force_resample=True)
                mask_data = resampled_mask.get_fdata().astype(bool)

                if mask_data.shape == sdbold_data.shape:
                    masked_vals = sdbold_data[mask_data]
                    record[seed] = np.mean(masked_vals)
                else:
                    print(f"⚠️ Shape mismatch for {seed} in {fname}, skipping.")
            except Exception as e:
                print(f"⚠️ Failed to process {seed} in {fname}: {e}")

        records.append(record)

df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"✅ Saved {output_csv} with {len(df)} subjects and {len(seed_masks)} seeds.")
