"""
Analyzes correlation between Functional Connectivity (FC) and cognitive scores (CVLT1, SES).
Author: Emre Pelzer
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

fc_dir = "features_fc"
cog_file = "cognitive_scores_normalized.csv"
output_file = "fc_correlation_results.csv"


# Load cognitive scores
df = pd.read_csv(cog_file)
cog_scores = df.set_index("subject")

# Collect mean FC values
fc_summary = []

for fname in os.listdir(fc_dir):
    if fname.endswith("_fc.npy"):
        parts = fname.replace(".npy", "").split("_")
        if len(parts) >= 3:
            sid = parts[0]
            seed = "_".join(parts[1:-1])  # seed names with underscores
            fc_map = np.load(os.path.join(fc_dir, fname))
            if fc_map.ndim == 3:
                mean_fc = np.mean(fc_map)
                fc_summary.append({"subject": sid, "seed": seed, "mean_fc": mean_fc})
            else:
                print(f"⚠️ Skipping {fname} due to unexpected shape: {fc_map.shape}")

fc_df = pd.DataFrame(fc_summary)

# Merge with cognition data
merged = fc_df.merge(cog_scores, left_on="subject", right_index=True)

# Correlations
results = []
for seed in merged["seed"].unique():
    seed_data = merged[merged["seed"] == seed]
    for cog_score in cog_scores.columns:
        r, p = pearsonr(seed_data["mean_fc"], seed_data[cog_score])
        results.append({
            "seed": seed,
            "cog_score": cog_score,
            "r": r,
            "p": p
        })

pd.DataFrame(results).to_csv(output_file, index=False)
print(f"\n✅ FC–Cognition correlations saved to: {output_file}")