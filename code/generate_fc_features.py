"""
Loops through subjects and generates FC feature CSVs using seed-based correlation analysis.
Author: Emre Pelzer
"""

import os
import numpy as np
import pandas as pd

feature_dir = "features_fc"
output_file = "fc_features.csv"

data = []

for file in os.listdir(feature_dir):
    if file.endswith("_fc.npy"):
        parts = file.split("_")
        subject = parts[0]
        region = "_".join(parts[1:-1])  # for underscore, such as: Hippocampus_left

        filepath = os.path.join(feature_dir, file)
        value = np.load(filepath)
        scalar_value = float(value) if value.size == 1 else float(np.mean(value))  # use mean if it is an array

        data.append({"subject": subject, "region": region, "fc": scalar_value})

# Create dataframe + pivot
df = pd.DataFrame(data)
df_pivot = df.pivot(index="subject", columns="region", values="fc").reset_index()

df_pivot.to_csv(output_file, index=False)
print(f"✅ Saved {output_file} with shape {df_pivot.shape}")