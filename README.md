# LeBron James High-Scoring Game Prediction

**Task:** Predict whether LeBron James records a high-scoring game (HSG ≥ 25 points) using pre-game tabular features derived from historical game logs.

---

## Project Overview

This project applies supervised machine learning to a binary classification task on a real-world sports dataset. Three model families are implemented and compared: Logistic Regression, K-Nearest Neighbors (KNN), and a custom Neural Network (MLP). The Neural Network was selected as the best-performing model based on macro-F1 score.

**Final test results (Neural Network):**

| Metric | Value |
|---|---|
| Test Accuracy | 65.9% |
| Macro F1-Score | 0.6051 |
| Decision Threshold | 0.385 |

> **Note on results:** The low scores are due to the models used (Logistic Regression, KNN, MLP) not being a great fit for this type of prediction task, not because of training errors. Predicting individual game performance from pre-game stats is inherently noisy, and these model types struggle to capture the patterns needed. This project is mainly **experimental** and was built to practice the ML pipeline end-to-end on a real-world dataset.

---

## Repository Structure

```
STINTSY-MCO-main/
├── main.ipynb                  # Combined deliverable notebook (full pipeline)
├── requirements.txt            # Python dependencies
│
├── notebooks/                  # Step-by-step development notebooks
│   ├── 1_Data_Cleaning.ipynb
│   ├── 2_Derived_Features.ipynb
│   ├── 3_eda.ipynb
│   ├── 4_logreg_reg.ipynb
│   ├── 5_knn.ipynb
│   └── 6_nn.ipynb
│
├── src/                        # Custom Python modules for the Neural Network
│   ├── nn_dataset.py           # Dataset loading, preprocessing, and train/valid/test splits
│   ├── nn_dataLoader.py        # Custom mini-batch data loader
│   ├── nn_model.py             # NeuralNetwork (MLP) architecture
│   └── nn_tuner.py             # Randomized hyperparameter search engine
│
├── data/
│   ├── raw/
│   │   └── LJ_Dataset_ORIG.csv           # Original scraped game log dataset
│   ├── processed/
│   │   ├── LJ_Dataset_NODERIV.csv        # Cleaned dataset (no derived features)
│   │   └── LJ_Dataset_DERIV.csv          # Final model dataset (with derived features)
│   └── splits/
│       ├── TRAIN_LJ_Dataset.csv          # 70% chronological training split
│       ├── VALID_LJ_Dataset.csv          # 15% validation split
│       ├── TEST_LJ_Dataset.csv           # 15% test split (without HSG label)
│       ├── TEST_LJ_Dataset_w_HSG.csv     # Test split with HSG label (for logreg eval)
│       └── TEST_LJ_Dataset_wout_HSG.csv  # Test split without HSG label
│
├── docs/
│   ├── specs.pdf                         # Official course MCO specification
│   ├── specs.txt                         # Plain-text copy of the specification
│   ├── HYPERPARAMETER_TUNING_EXPLANATION.md  # Detailed tuning methodology notes
│   └── NEURAL_NETWORK_POSTER_NOTES.md        # Poster draft and oral defense prep
│
└── archive/                    # Draft notebooks, submission zip, and prior working copy
```

---

## Dataset

**Source:** LeBron James game log data (publicly available, scraped from basketball-reference.com).  
**Target variable:** `HSG` — binary label, 1 if points ≥ 25, 0 otherwise.  
**Split strategy:** Chronological (no shuffling to prevent data leakage):

- Train: 70%
- Validation: 15%
- Test: 15%

The dataset undergoes multi-stage preprocessing: raw → cleaned (`NODERIV`) → feature-engineered (`DERIV`) → chronological splits.

---

## Models Implemented

| Model | Description |
|---|---|
| Logistic Regression | L1/L2 regularization, threshold tuned on validation set |
| K-Nearest Neighbors | Distance-based classifier, k tuned via validation macro-F1 |
| Neural Network (MLP) | Feedforward MLP with BatchNorm, Dropout, AdamW, and ReduceLROnPlateau |

All models use the same chronological train/validation/test splits. Final model selection uses macro-F1 as the primary metric, with accuracy and balanced accuracy as tie-breakers.

---

## Neural Network Architecture

- **Input:** 11 processed tabular features
- **Hidden layers:** [192, 96] units
- **Activation:** ReLU
- **Regularization:** BatchNorm1d, Dropout (p ≈ 0.058), weight decay
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` for class imbalance
- **Optimizer:** AdamW
- **Scheduler:** ReduceLROnPlateau on validation loss
- **Threshold:** Tuned on validation set (grid: 0.10–0.90, 161 points); selected by macro-F1

Hyperparameter search (Step 9 in `main.ipynb`) uses a validation-first randomized search with focused and unfocused sampling branches. See [docs/HYPERPARAMETER_TUNING_EXPLANATION.md](docs/HYPERPARAMETER_TUNING_EXPLANATION.md) for full methodology.

---

## Setup and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter from the project root

```bash
jupyter notebook
```

> **Important:** Always launch from the project root directory. All notebooks use paths relative to the root (e.g., `data/splits/TRAIN_LJ_Dataset.csv`).

### 3. Run the pipeline

**Recommended:** Open `main.ipynb` for the complete end-to-end pipeline.

Alternatively, run the numbered notebooks in order:

| Notebook | Description |
|---|---|
| `notebooks/1_Data_Cleaning.ipynb` | Load raw data, fix types, remove invalid rows |
| `notebooks/2_Derived_Features.ipynb` | Engineer rolling average and contextual features |
| `notebooks/3_eda.ipynb` | Exploratory data analysis and chronological splits |
| `notebooks/4_logreg_reg.ipynb` | Logistic Regression with regularization and tuning |
| `notebooks/5_knn.ipynb` | KNN implementation and evaluation |
| `notebooks/6_nn.ipynb` | Neural Network training, tuning, and evaluation |

---
