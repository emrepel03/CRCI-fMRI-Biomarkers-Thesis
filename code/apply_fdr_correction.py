"""
Applies False Discovery Rate (FDR) correction to p-values from correlation or group analysis.
Author: Emre Pelzer
"""

import pandas as pd
from statsmodels.stats.multitest import multipletests
import os

# Files for FDR
files = ["fc_correlation_results.csv", "sdbold_correlation_results.csv"]

for file in files:
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        continue

    df = pd.read_csv(file)

    if "p" not in df.columns:
        print(f"❌ No 'p' column found in {file} — skipping FDR correction.")
        continue

    # FDR correction
    pvals = df["p"].values
    rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    # Add results to dataframe
    df["p_fdr"] = pvals_corrected
    df["significant_fdr"] = rejected

    # Save
    output_file = file.replace(".csv", "_fdr.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ FDR correction applied and saved to {output_file}")