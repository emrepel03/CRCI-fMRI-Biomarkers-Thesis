# CRCI Thesis Code – FC and SD-BOLD Analysis

This repository contains all code for the analysis of **Functional Connectivity (FC)** and **BOLD Signal Variability (SD-BOLD)** in resting-state fMRI data. It evaluates how these neural features relate to cognitive performance (memory and executive function) in healthy female participants, forming a baseline for identifying biomarkers of cancer-related cognitive impairment (CRCI).

Used for the bachelor thesis:  
**"Functional Connectivity and BOLD Variability as Neural Biomarkers for Cancer-Related Cognitive Impairment"**  
Author: Emre Pelzer | Maastricht University | 2025

---

## What This Code Does

1. **Preprocesses raw rs-fMRI** in native functional space (motion correction, 6mm smoothing).
2. **Extracts Functional Connectivity (FC)** using seed-based correlation from 13 brain regions (DMN, FPN, SN, hippocampus).
3. **Extracts SD-BOLD** by computing voxelwise signal variability.
4. **Performs cognitive analyses**:
   - Correlations with CVLT1 (memory) and SES (executive function)
   - Group comparisons between high vs. low memory performers
5. **Outputs**:
   - Per-subject CSV files with extracted features  
   - Statistical test results  
   - Figures and summary plots

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your data
- Download from: [OpenNeuro ds004796 – PEARL-Neuro Dataset](https://openneuro.org/datasets/ds004796/versions/1.0.9)
- Select female participants with `task-rest_dir-PA_bold.nii.gz`
- Organize data in BIDS format under `/data/`

### 3. Preprocess
```bash
python generate_sdbold_per_subject_resampled.py
```

### 4. Extract features
```bash
python generate_fc_features.py
```

### 5. Analyze correlations
```bash
python analyze_fc_cognition.py
python correlate_sdbold_scores.py
```

### 6. Group comparisons
```bash
python analyze_fc_cvlt_groups.py
python analyze_sdbold_cvlt_groups.py
```

---

## Folder Structure

- `/code/`: all scripts  
- `/data/`: raw data in BIDS format (publicly available via OpenNeuro)

---

## Notes

- Preprocessing is done **without T1w normalization**
- SD-BOLD is computed voxelwise and averaged per ROI
- FC values are Fisher z-transformed
- Only **female participants** included to match CRCI population
- Cognitive scores (CVLT1, SES) are z-normalized

---

## License

Creative Commons BY-NC-ND 4.0 – see LICENSE file for details.
If reusing thesis text or figures, please credit the author.

---

## Contact

**Emre Pelzer**  
B.Sc. Data Science & AI – Maastricht University
Mail: emrepel03@gmail.com