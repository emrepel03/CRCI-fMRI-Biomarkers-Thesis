"""
Analyzes correlation between SD-BOLD variability and cognitive scores (CVLT1, SES).
Author: Emre Pelzer
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiSpheresMasker
from scipy.stats import pearsonr

sdbold_dir = "features_sdbold"
scores_file = "cognitive_scores_normalized.csv"

# Seed coordinates (same as in extraction)
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
    "Hippocampus_left": [-24, -18, -20],
    "Hippocampus_right": [24, -18, -20]
}

scores_df = pd.read_csv(scores_file)
subjects = scores_df["subject"].tolist()
cog_scores = scores_df.drop(columns=["subject"])

# Prepare matrix to store SD-BOLD values per seed
sdbold_values = {seed: [] for seed in seeds}
valid_subjects = []

# Extract mean SD-BOLD per seed
for subject in subjects:
    print(f"🔄 Processing {subject}...")
    filepath = os.path.join(sdbold_dir, f"{subject}_sdbold.nii.gz")
    if not os.path.exists(filepath):
        print(f"⚠️ Missing file: {filepath}")
        continue

    try:
        img = nib.load(filepath)
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        continue

    valid_subjects.append(subject)

    for name, coord in seeds.items():
        masker = NiftiSpheresMasker([coord], radius=6, detrend=False, standardize=False)
        try:
            ts = masker.fit_transform(img)
            mean_val = np.mean(ts)
        except Exception as e:
            print(f"❌ Failed masker for {subject} at {name}: {e}")
            mean_val = np.nan
        sdbold_values[name].append(mean_val)

sdbold_df = pd.DataFrame(sdbold_values)
sdbold_df.insert(0, "subject", valid_subjects)

merged = pd.merge(sdbold_df, scores_df, on="subject")

# Correlations
correlation_results = []
for seed in seeds:
    for score in cog_scores.columns:
        x = merged[seed]
        y = merged[score]
        if x.isnull().any() or y.isnull().any():
            corr = np.nan
            pval = np.nan
        else:
            corr, pval = pearsonr(x, y)
        correlation_results.append({
            "seed": seed,
            "score": score,
            "correlation": corr,
            "p_value": pval
        })

results_df = pd.DataFrame(correlation_results)

output_path = os.path.join(os.path.dirname(__file__), "sdbold_correlation_results.csv")
results_df.to_csv(output_path, index=False)
print("✅ Correlations saved to:", output_path)
