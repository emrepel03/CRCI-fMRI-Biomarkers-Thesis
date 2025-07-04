"""
Performs group comparison of SD-BOLD values between high and low CVLT1 (memory) performers.
Author: Emre Pelzer
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

feature_dir = "features_sdbold"
score_file = "cognitive_scores_normalized.csv"
cvlt_cols = ["CVLT1", "CVLT2", "CVLT3", "CVLT4", "CVLT5", "CVLT6", "CVLT7", "CVLT8", "CVLT9", "CVLT10", "CVLT11", "CVLT12", "CVLT13"]
output_csv = "sdbold_cvlt_group_comparison.csv"
output_dir = "sdbold_cvlt_plots"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(score_file)
df["CVLT_total"] = df[cvlt_cols].sum(axis=1)

# Median split
median_cvlt = df["CVLT_total"].median()
df["cvlt_group"] = ["high" if score >= median_cvlt else "low" for score in df["CVLT_total"]]

results = []

# Analysis on each region
for fname in os.listdir(feature_dir):
    if not fname.endswith("_sdbold.nii.gz"):
        continue

    subject = fname.split("_")[0]
    sdbold_path = os.path.join(feature_dir, fname)

    if subject not in df["subject"].values:
        continue

    try:
        data = nib.load(sdbold_path).get_fdata()
        mean_val = np.mean(data[data > 0])  # ignore empty voxels

        group = df[df["subject"] == subject]["cvlt_group"].values[0]
        results.append((subject, group, mean_val))

    except Exception as e:
        print(f"⚠️ Skipped {subject}: {e}")

res_df = pd.DataFrame(results, columns=["subject", "group", "mean_sdbold"])

# Split groups
low_vals = res_df[res_df["group"] == "low"]["mean_sdbold"]
high_vals = res_df[res_df["group"] == "high"]["mean_sdbold"]

tval, pval = ttest_ind(low_vals, high_vals, equal_var=False)

# Save result in CSV
summary = {
    "t_statistic": tval,
    "p_value": pval,
    "mean_low": low_vals.mean(),
    "mean_high": high_vals.mean(),
    "n_low": len(low_vals),
    "n_high": len(high_vals)
}
pd.DataFrame([summary]).to_csv(output_csv, index=False)
print(f"✅ Saved result to {output_csv}")

# Boxplot
plt.boxplot([low_vals, high_vals], labels=["Low CVLT", "High CVLT"])
plt.ylabel("Mean SD-BOLD")
plt.title("SD-BOLD by CVLT Group")
plt.savefig(os.path.join(output_dir, "sdbold_cvlt_boxplot.png"))
plt.show()
