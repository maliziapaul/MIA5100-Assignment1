# MIA5100 — Assignment 1

**Course:** MIA5100  
**Student:** Paul Malizia  
**Email:** pmali097@uottawa.ca

---

## Overview

This assignment covers four parts using two real-world datasets:

| Part | Topic | Marks | Dataset |
|------|-------|-------|---------|
| 1 | Data Wrangling & Feature Engineering | 30 | MedicalCentre.csv |
| 2 | Model Development & Evaluation (Naïve Bayes) | 20 | MedicalCentre.csv |
| 3 | Model Comparison (NB vs Ensemble) | 30 | MedicalCentre.csv |
| 4 | Unsupervised Learning (K-Means + DBSCAN) | 20 | housing.csv |

---

## Datasets

**`MedicalCentre.csv`** — 110,527 medical appointment records with features including patient demographics, medical conditions, scheduling dates, and whether the patient showed up (`No-show`).

**`housing.csv`** — 20,640 California housing block records with median income, house age, average rooms, population, and median house value.

---

## Notebook

**`Assignment1_DataWrangling.ipynb`** — single Jupyter notebook covering all parts (95 cells).

### Part 1 — Data Wrangling & Feature Engineering
- Handle missing values, duplicates, and irrelevant attributes
- Boxplot outlier detection
- Remove negative Age values; create `AgeGroup` bins (1–10)
- Min-Max normalize Age → `Age_normalized`; drop original
- Derive `AwaitingTime` from date difference; convert negatives to positive
- Bin into `AwaitingTimeGroup` (1–5); drop original
- Decompose `ScheduledDay` and `AppointmentDay` into date components
- Label-encode categorical features (Gender, Neighbourhood, No-show)
- Variability comparison (variance, std dev, coefficient of variation)

### Part 2 — Model Development & Evaluation
- Gaussian Naïve Bayes classifier on a stratified 70/30 train-test split
- Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- Overfitting check: train vs test performance comparison
- `GridSearchCV` over `var_smoothing` with 5-fold stratified cross-validation

### Part 3 — Model Comparison
- Random Forest (100 trees) and XGBoost (100 trees, lr=0.1)  trained on the same split
- Side-by-side comparison: Accuracy, Sensitivity, Specificity, F1, ROC-AUC
- Best/worst model identified per criterion
- ROC curves for all four models on a single graph with AUC discrimination labels

### Part 4 — Unsupervised Learning
**Option (a) — K-Means Clustering (`housing.csv`)**
- Features: `MedInc`, `Latitude`, `Longitude` (StandardScaler applied)
- k = 6 clusters; geographic scatter plots + income distribution per cluster

**Option (b) — DBSCAN Theoretical Analysis**
- Similarity matrix (5×5), similarity threshold = 0.8, MinPts = 2
- Step-by-step neighbourhood analysis → core, border, and noise point classification

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
ipykernel
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost ipykernel
```

---

## Running the Notebook

1. Clone the repo and open the folder in VS Code (or Jupyter Lab)
2. Select the **Python 3** kernel
3. Run all cells top-to-bottom (**Shift+Enter** or **Run All**)

> Both CSV files must be in the same directory as the notebook.
