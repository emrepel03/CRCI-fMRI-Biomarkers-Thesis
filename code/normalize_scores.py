"""
Z-normalizes cognitive test scores (e.g., CVLT1, SES) across participants.
Author: Emre Pelzer
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("cognitive_scores_clean.csv")

subjects = df["subject"]
features = df.drop(columns=["subject"])

scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
normalized_df.insert(0, "subject", subjects)

normalized_df.to_csv("cognitive_scores_normalized.csv", index=False)
print("✅ Normalized scores saved to cognitive_scores_normalized.csv")