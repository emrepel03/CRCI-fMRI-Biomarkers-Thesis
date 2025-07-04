"""
Performs group comparison of FC features between high and low CVLT1 (memory) performers.
Author: Emre Pelzer
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

feature_dir = "features_fc"
score_file = "cognitive_scores_normalized.csv"
cvlt_cols = ["CVLT1", "CVLT2", "CVLT3", "CVLT4", "CVLT5", "CVLT6", "CVLT7", "CVLT8", "CVLT9", "CVLT10"]
output_csv = "fc_cvlt_group_comparison.csv"
output_dir = "fc_cvlt_plots"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(score_file)
df["CVLT_total"] = df[cvlt_cols].sum(axis=1)

# Median split
median_cvlt = df["CVLT_total"].median()
df["cvlt_group"] = ["high" if score >= median_cvlt else "low" for score in df["CVLT_total"]]

results = []

# Analysis
for fname in os.listdir(feature_dir):
    if not fname.endswith("_fc.npy"):
        continue

    parts = fname.replace(".npy", "").split("_")
    if len(parts) < 3:
        continue

    subject = parts[0]
    seed = "_".join(parts[1:-1])  # handles seeds with underscores
    fc_path = os.path.join(feature_dir, fname)

    if subject not in df["subject"].values:
        continue

    try:
        data = np.load(fc_path)
        mean_val = np.mean(data[np.isfinite(data)])

        group = df[df["subject"] == subject]["cvlt_group"].values[0]
        results.append((seed, subject, group, mean_val))

    except Exception as e:
        print(f"⚠️ Skipped {fname}: {e}")


res_df = pd.DataFrame(results, columns=["seed", "subject", "group", "mean_fc"])
summary_rows = []

for seed in res_df["seed"].unique():
    seed_data = res_df[res_df["seed"] == seed]
    low_vals = seed_data[seed_data["group"] == "low"]["mean_fc"]
    high_vals = seed_data[seed_data["group"] == "high"]["mean_fc"]

    if len(low_vals) >= 2 and len(high_vals) >= 2:
        tval, pval = ttest_ind(low_vals, high_vals, equal_var=False)
        summary = {
            "seed": seed,
            "t_statistic": tval,
            "p_value": pval,
            "mean_low": low_vals.mean(),
            "mean_high": high_vals.mean(),
            "n_low": len(low_vals),
            "n_high": len(high_vals)
        }
        summary_rows.append(summary)

        # Boxplot
        plt.boxplot([low_vals, high_vals], labels=["Low CVLT", "High CVLT"])
        plt.ylabel("Mean FC")
        plt.title(f"FC by CVLT Group: {seed}")
        plot_path = os.path.join(output_dir, f"{seed}_fc_boxplot.png")
        plt.savefig(plot_path)
        plt.close()

pd.DataFrame(summary_rows).to_csv(output_csv, index=False)
print(f"\n✅ Group comparisons saved to: {output_csv}")
