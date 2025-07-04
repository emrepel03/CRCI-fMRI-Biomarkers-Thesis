"""
Contains helper functions to generate plots for correlation and group analysis results.
Author: Emre Pelzer
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
os.makedirs("output", exist_ok=True)

# Load data
fc_corr = pd.read_csv("fc_correlation_results_fdr.csv")
fc_group = pd.read_csv("fc_cvlt_group_comparison.csv")
sdbold_corr = pd.read_csv("sdbold_correlation_results_fdr.csv")

# Heatmap of FC correlations (filtered by FDR-significance)
print("FC correlation columns:", fc_corr.columns.tolist())
fc_pivot = fc_corr.pivot(index="seed", columns="cog_score", values="r")
print(fc_pivot)  # Debugging output
fc_pivot = fc_pivot.fillna(0)
plt.figure(figsize=(10, 6))
sns.heatmap(fc_pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Functional Connectivity vs. Cognitive Scores (r-values)")
plt.tight_layout()
plt.savefig("output/fc_heatmap.png", dpi=300)
plt.close()

# Boxplot: Group difference in FC (hippocampus)
fc_group["cvlt_group"] = np.where(fc_group["mean_low"] < fc_group["mean_high"], "low", "high")
fc_group["hippocampus_fc"] = (fc_group["mean_low"] + fc_group["mean_high"]) / 2

sns.boxplot(data=fc_group, x="cvlt_group", y="hippocampus_fc")
plt.title("FC: Left Hippocampus in Low vs High CVLT")
plt.tight_layout()
plt.savefig("output/fc_hippocampus_group_boxplot.png", dpi=300)
plt.close()

# Scatter plot of significant FC correlation (right hippocampus vs CVLT1)
sig_fc = fc_corr[(fc_corr["seed"] == "Hippocampus_right") & (fc_corr["cog_score"] == "CVLT1")]
if not sig_fc.empty:
    try:
        fc_values = pd.read_csv("fc_features.csv")  
        cog_scores = pd.read_csv("cognitive_scores_clean.csv") 

        merged_df = pd.merge(fc_values, cog_scores, on="subject")

        plt.figure(figsize=(6, 5))
        sns.regplot(data=merged_df, x="Hippocampus_right", y="CVLT1", scatter_kws={'s': 50})
        plt.title("Right Hippocampus FC vs CVLT1")
        plt.tight_layout()
        plt.savefig("output/fc_hippocampus_vs_cvlt.png", dpi=300)
        plt.close()
    except FileNotFoundError:
        print("Required per-subject data for FC scatter plot not found.")

# SD-BOLD group boxplot using per-subject values for AINS_left
try:
    sdbold_values = pd.read_csv("sdbold_per_subject.csv")  # wide format: subject, seed1, seed2, etc
    cog_scores = pd.read_csv("cognitive_scores_clean.csv")

    # Long format
    sdbold_long = sdbold_values.melt(id_vars="subject", var_name="seed", value_name="value")

    ains_left_data = sdbold_long[sdbold_long["seed"] == "AINS_left"]
    merged_sd = pd.merge(ains_left_data, cog_scores, on="subject")
    median_cvlt1 = merged_sd["CVLT1"].median()
    merged_sd["cvlt_group"] = np.where(merged_sd["CVLT1"] <= median_cvlt1, "low", "high")

    plt.figure(figsize=(6, 5))
    sns.boxplot(data=merged_sd, x="cvlt_group", y="value")
    plt.title("SD-BOLD AINS_left in Low vs High CVLT1")
    plt.tight_layout()
    plt.savefig("output/sdbold_ains_left_group_boxplot.png", dpi=300)
    plt.close()
except FileNotFoundError:
    print("Required per-subject SD-BOLD data not found.")

# Scatter plot of significant SD-BOLD correlation (insula)
if "cog_score" in sdbold_corr.columns:
    sig_sd = sdbold_corr[(sdbold_corr["seed"] == "AINS_left") & (sdbold_corr["cog_score"] == "CVLT1")]
else:
    sig_sd = pd.DataFrame()
if not sig_sd.empty:
    try:
        sdbold_values = pd.read_csv("sdbold_per_subject.csv")  # Contains subject, seed, value
        cog_scores = pd.read_csv("cognitive_scores_clean.csv") 

        merged_df = pd.merge(sdbold_values, cog_scores, on="subject")
        merged_df = merged_df[merged_df["seed"] == "AINS_left"]

        plt.figure(figsize=(6, 5))
        sns.regplot(data=merged_df, x="value", y="CVLT1", scatter_kws={'s': 50})
        plt.title("SD-BOLD AINS Left vs CVLT1")
        plt.tight_layout()
        plt.savefig("output/sdbold_ains_vs_cvlt.png", dpi=300)
        plt.close()
    except FileNotFoundError:
        print("Required per-subject data for SD-BOLD scatter plot not found.")

# Scatter and Boxplots from FC Features (hypothesis)
try:
    features_fc = pd.read_csv("fc_features.csv")
    cog = pd.read_csv("cognitive_scores_normalized.csv")

    merged = pd.merge(features_fc, cog, on="subject")

    # H2: Scatter plot - Hippocampus_right vs CVLT1
    plt.figure(figsize=(6, 5))
    sns.regplot(data=merged, x="Hippocampus_right", y="CVLT1", scatter_kws={'s': 50})
    plt.title("Hippocampus Right FC vs CVLT1")
    plt.tight_layout()
    plt.savefig("output/fc_hippocampus_right_vs_cvlt1.png", dpi=300)
    plt.close()

    # H3: Boxplot - AINS_left vs CVLT1 group (low vs high performing groups)
    median_cvlt1 = merged["CVLT1"].median()
    merged["cvlt_group"] = np.where(merged["CVLT1"] <= median_cvlt1, "low", "high")

    plt.figure(figsize=(6, 5))
    sns.boxplot(data=merged, x="cvlt_group", y="AINS_left")
    plt.title("AINS Left FC in Low vs High CVLT1")
    plt.tight_layout()
    plt.savefig("output/fc_ains_left_group_boxplot.png", dpi=300)
    plt.close()
except FileNotFoundError:
    print("Required FC features or cognitive data not found — skipping hypothesis-based plots.")

os.makedirs("output", exist_ok=True)