"""
Extracts relevant cognitive scores (CVLT1, SES, BDI) from participant data files.
Author: Emre Pelzer
"""

import pandas as pd

df = pd.read_csv("cognitive_scores.csv", header=None, delimiter="\t", engine="python")

# Column indices:
subject_col = 0
bdi_col = 23
ses_col = 24
cvlt_cols = list(range(47, 60))  # CVLT1–CVLT13

# New dataframe with only the columns selected
columns_to_keep = [subject_col, bdi_col, ses_col] + cvlt_cols
df_clean = df[columns_to_keep]

# Rename columns for clarity
df_clean.columns = ["subject", "BDI", "SES"] + [f"CVLT{i}" for i in range(1, 14)]

df_clean.to_csv("cognitive_scores_clean.csv", index=False)
print("✅ Saved cleaned file as cognitive_scores_clean.csv")