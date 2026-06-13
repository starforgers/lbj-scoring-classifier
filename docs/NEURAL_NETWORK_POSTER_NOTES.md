# Digital Poster Draft Notes - Neural Network Model (Detailed Version)

## 1. Project Context and Poster Goal

This document is a long-form draft of key notes for the digital poster, focused on our selected best model: the Neural Network.

Based on the course specs, the poster should clearly communicate:

1. The project goal.
2. Dataset description.
3. Best model and architecture details.
4. Key results and findings.
5. Any unique strengths of the approach.

This draft is intentionally text-heavy so the team can cut and compress later into poster-friendly bullets and visuals.

---

## 2. Problem Statement and Motivation

### Supervised learning task

Predict whether LeBron James will have a high-scoring game (HSG = 1) or not (HSG = 0) based on pre-game and historical features.

In short:

Predict HSG based on structured tabular features derived from game logs.

### Why this is meaningful

The task is interesting because it combines sports intuition with machine learning decision-making under class imbalance and overlapping classes. It also allows direct comparison between:

1. Linear model behavior (Logistic Regression).
2. Distance-based model behavior (KNN).
3. Nonlinear representation learning (Neural Network).

The Neural Network was selected as best based on evaluation metrics and final model comparison rules.

---

## 3. Dataset and Target Definition

### Source files used in the workflow

1. Raw source and cleaned/derived files in the repository.
2. Final split files used by the Neural Network section:
   - TRAIN_LJ_Dataset.csv
   - VALID_LJ_Dataset.csv
   - TEST_LJ_Dataset.csv

### Target variable

HSG is a binary target:

1. HSG = 1 if points scored >= 25.
2. HSG = 0 otherwise.

### Split strategy

Chronological split was used earlier in the notebook pipeline:

1. Train: 70%
2. Validation: 15%
3. Test: 15%

This is important for leakage prevention and realistic evaluation over time.

---

## 4. End-to-End Neural Network Pipeline (Notebook Steps)

Neural Network section is organized into a reproducible sequence in the notebook.

### Step 1: Set seed and load data

1. Fixed seed for random, numpy, and torch.
2. Loaded train/valid/test splits through custom function load_tabular_splits.
3. Printed dimensions for quick sanity checks.

### Step 2: Configure training hyperparameters

1. Baseline training hyperparameters were set.
2. Threshold grid was prepared for validation-based threshold tuning.
3. Positive and negative class counts were computed.
4. pos_weight for BCEWithLogitsLoss was derived as neg_count / pos_count.
5. Device chosen automatically (cuda if available, otherwise cpu).

### Step 3: Build mini-batch loaders

1. Created custom DataLoader objects for train, valid, and test splits.
2. Batching enabled efficient training and evaluation.

### Step 4: Initialize Neural Network

1. Constructed NeuralNetwork with hidden layers [192, 96], activation relu, and dropout around 0.0578.
2. Created network layers and initialized weights with fixed seed.
3. Moved model to selected device.
4. Configured criterion = BCEWithLogitsLoss with class weighting.
5. Configured optimizer = AdamW.
6. Configured scheduler = ReduceLROnPlateau.

### Step 5: Define epoch runner

1. Unified function for train and eval passes.
2. Computes batch-wise logits and loss.
3. Handles gradient updates only in train mode.
4. Returns epoch average loss.

### Step 6: Baseline training with early stopping

1. Trains for up to max_epochs.
2. Tracks best validation loss checkpoint.
3. Applies learning-rate scheduling using validation loss.
4. Stops early after patience epochs without improvement.
5. Restores best checkpoint.

### Step 7: Training diagnostics

1. Plots train and validation BCE loss curves.
2. Used to inspect convergence and overfitting behavior.

### Step 8: Baseline evaluation and threshold tuning

1. Predicts validation probabilities.
2. Searches best threshold on validation set using Macro-F1 objective.
3. Applies selected threshold to test probabilities.
4. Reports test accuracy, balanced accuracy, and macro-F1.

### Step 8A: Baseline confusion matrix

1. Visualizes test confusion matrix for error pattern analysis.
2. Confirms class-specific strengths and weaknesses.

### Step 9: Hyperparameter tuning (validation-first)

1. Runs randomized search via separate tuning engine module.
2. Uses focused + unfocused sampling for exploration/exploitation balance.
3. Per trial:
   - trains model with early stopping,
   - tunes threshold on validation,
   - logs trial metrics and config.
4. Ranks by validation Macro-F1.
5. Preserves Section 8 comparison variables to avoid accidental overwrite.

### Step 9A and 9B: Tuning diagnostics

1. Validation ranking plot.
2. Validation-ranked trial table and best configuration summary.

---

## 5. Custom Modules Used and Their Roles

### nn_dataset.py

Primary role: dataset loading, normalization, encoding, and scaling.

What it does:

1. Defines TabularSplits dataclass containing train/valid/test matrices and labels.
2. Normalizes date column to ordinal format when present.
3. Splits features and target cleanly.
4. Applies one-hot encoding via get_dummies.
5. Aligns valid/test columns to train feature schema.
6. Applies StandardScaler fit on train, then transform valid/test.
7. Returns float32 arrays and metadata.

Why this matters:

1. Ensures consistent feature space across splits.
2. Prevents train-test leakage in scaling.
3. Makes arrays directly usable by PyTorch.

### nn_dataLoader.py

Primary role: simple custom mini-batch loader.

What it does:

1. Stores X, y, batch_size.
2. Shuffles indices in train mode.
3. Keeps deterministic index order in test mode.
4. Returns batches as list of arrays.

Why this matters:

1. Gives explicit control over mini-batch behavior.
2. Keeps implementation transparent for oral defense.

### nn_model.py

Primary role: defines the NeuralNetwork architecture and forward behavior.

Key model structure:

1. Hidden blocks repeat:
   - Linear
   - BatchNorm1d
   - Activation (relu or tanh)
   - Dropout
2. Final layer is Linear with output size 1.
3. Forward pass returns:
   - logits (raw score)
   - probabilities = sigmoid(logits)

Important correction applied:

1. Final architectural sigmoid layer was removed from the Sequential stack.
2. This aligns correctly with BCEWithLogitsLoss, which already handles logits numerically stably.
3. Sigmoid is only used for probability output and thresholding.

### nn_tuner.py

Primary role: reusable hyperparameter tuning execution engine.

Functions included:

1. run_epoch
2. predict_probabilities
3. find_best_threshold_f1
4. run_tuning_search

Why this matters:

1. Separates tuning execution logic from notebook strategy logic.
2. Keeps notebook readable and focused on experiment design.
3. Enables repeatable trial search with consistent metric logic.

---

## 6. Neural Network Architecture Details

Baseline architecture used in the selected NN section:

1. Input dimension: number of processed tabular features (11 in the current run).
2. Hidden layer 1: 192 units.
3. Hidden layer 2: 96 units.
4. Activation: ReLU.
5. Dropout probability: 0.0578317383875907.
6. BatchNorm used after each hidden linear layer.
7. Output layer: 1 unit (logit).

Conceptually, this is a feedforward multilayer perceptron (MLP) for binary classification.

### Forward computation summary

1. Feature vector passes through hidden nonlinear transformations.
2. Final linear output is a logit z.
3. Probability is computed as sigmoid(z).
4. Classification threshold is tuned on validation set instead of fixed at 0.5.

---

## 7. Training Objective, Loss, and Optimization

### Loss function

BCEWithLogitsLoss with positive-class weighting.

Reason:

1. Binary task.
2. Better numerical stability than sigmoid + BCELoss done separately.
3. Supports class imbalance weighting directly with pos_weight.

### Class imbalance handling

1. pos_weight = number of negatives / number of positives from train split.
2. This increases penalty for minority positive-class errors during training.

### Optimizer

AdamW.

Reason:

1. Works well for tabular MLP optimization.
2. Decoupled weight decay improves regularization control.

### Scheduler

ReduceLROnPlateau on validation loss.

Configured to reduce LR when validation improvement plateaus.

### Early stopping

1. Track best validation loss checkpoint.
2. Stop when no improvement for patience epochs.
3. Restore best checkpoint before evaluation.

This keeps training stable and reduces overfitting risk.

---

## 8. Threshold Tuning and Why It Was Important

Instead of locking threshold = 0.5, threshold was tuned on validation probabilities.

Threshold grid:

1. 0.10 to 0.90
2. 161 evenly spaced points (step 0.005)

Selection rule in threshold search:

1. Maximize Macro-F1.
2. Tie-break with balanced accuracy.
3. If still tied, tie-break with accuracy.

Why this matters:

1. Class distributions and score calibration can make 0.5 suboptimal.
2. Macro-F1 objective improves class-balance quality, not just majority accuracy.

---

## 9. Hyperparameter Tuning Methodology (Step 9)

### High-level strategy

Validation-first randomized search using a mixed sampler:

1. Focused mode (90%): exploit promising regions via branch-specific priors.
2. Unfocused mode (10%): preserve broad exploration.

### Trial budget and controls (current notebook run)

1. max_trials = 50
2. trial_max_epochs = 320
3. trial_patience = 40
4. trial_threshold_grid = linspace(0.10, 0.90, 161)
5. progress logging every 5 trials

### Search dimensions

1. seed
2. hidden architecture
3. activation
4. batch size
5. dropout
6. learning rate (log-uniform)
7. weight decay (log-uniform)
8. decision threshold (per-trial tuned)

### Focused branch hypotheses

Branch 0:

1. Medium ReLU stacks
2. Light regularization
3. Moderate LR/WD ranges

Branch 1:

1. Wider ReLU stacks
2. Stronger dropout/weight decay
3. Regularized high-capacity hypothesis

Branch 2:

1. Tanh high-capacity alternatives
2. Wider optimization ranges

Branch 3 fallback:

1. Conservative ReLU settings near previously strong region

### Why log-uniform for LR and WD

1. Useful values span multiple orders of magnitude.
2. Log sampling prevents bias toward large values and improves search coverage.

### Per-trial workflow

1. Sample config.
2. Reset seeds.
3. Build trial loaders with sampled batch size.
4. Build and initialize model.
5. Train with AdamW + ReduceLROnPlateau + early stopping.
6. Restore best validation-loss checkpoint.
7. Tune threshold on validation set.
8. Store trial metrics and full model state dictionary.

### Trial ranking

1. Rank by valid_f1 (descending).
2. Best trial selected by highest validation Macro-F1.

### Experimental safety measure

Before Step 9 tuning, key Section 8 comparison variables were backed up and restored afterward so benchmark comparison outputs remain stable.

---

## 10. Final Neural Network Results (Selected Model)

From notebook conclusions and outputs:

1. Test Accuracy: 0.6589 (65.9%)
2. Test Macro-F1: 0.6051
3. Operating threshold: 0.385
4. Confusion matrix:
   - TN = 31
   - FP = 45
   - FN = 28
   - TP = 110

Interpretation:

1. Model identifies HSG (class 1) strongly.
2. Class 0 recall is lower, but still materially recovered compared with weaker baselines.
3. Macro-F1 outperformed classical models in the same project, making NN the selected best model.

---

## 11. Why the Neural Network Was Chosen as Best

### Evaluation policy used in the project

Model comparison in Section 8 followed:

1. Primary metric: F1 score.
2. Tie-breaker 1: Accuracy.
3. Tie-breaker 2: Balanced Accuracy.

### Why NN won

1. Best macro-level class-balanced performance among tested models.
2. Better nonlinear feature interaction capture than linear baseline.
3. More competitive class-sensitive behavior than KNN under this feature space.
4. Stable training with regularization and threshold tuning.

---

## 12. Suggested Poster Sections (Content Blueprint)

Use these as blocks for your final layout.

### A. Problem and objective

1. Predict LeBron high-scoring games (>=25 points).
2. Binary classification on tabular historical game features.

### B. Data pipeline

1. Chronological splitting (70/15/15).
2. Feature engineering and preprocessing.
3. Standardization and target definition.

### C. Model shortlist

1. Logistic Regression
2. K-Nearest Neighbors
3. Neural Network (selected best)

### D. Neural Network architecture

1. MLP with hidden [192, 96]
2. ReLU + BatchNorm + Dropout
3. Output logit + sigmoid for probabilities
4. BCEWithLogitsLoss + pos_weight

### E. Training and tuning strategy

1. AdamW + ReduceLROnPlateau
2. Early stopping by validation loss
3. Validation-based threshold tuning
4. Step 9 randomized search with focused/unfocused sampling

### F. Key metrics and confusion matrix

1. Test Accuracy 65.9%
2. Macro-F1 0.6051
3. Threshold 0.385
4. Confusion matrix counts

### G. Main findings

1. Neural network best balances predictive quality and class sensitivity.
2. Validation-first tuning and threshold optimization were decisive.

---

## 13. Oral Defense Prep Notes (Useful Talking Points)

### Explain how the neural network works in simple terms

1. It transforms input features through stacked weighted layers.
2. Hidden layers learn nonlinear feature interactions.
3. Final logit represents confidence before probability conversion.
4. Sigmoid maps logit to probability for thresholded classification.

### Explain why BCEWithLogitsLoss was used

1. Numerically stable for binary classification.
2. Correctly works with raw logits.
3. Allows class weighting through pos_weight.

### Explain how overfitting was controlled

1. Dropout in hidden layers.
2. Weight decay in AdamW.
3. LR scheduling on validation plateaus.
4. Early stopping and checkpoint restore.

### Explain why threshold tuning was necessary

1. Default 0.5 is not always optimal under class imbalance.
2. Validation-tuned threshold improved Macro-F1 and class balance.

### Explain hyperparameter tuning design choice

1. Focused branches encoded informed hypotheses.
2. Unfocused trials preserved discovery.
3. Selection stayed strictly validation-first to avoid test leakage.

---

## 14. Limitations and Future Improvements

### Current limitations

1. Class 0 recall remains lower than class 1.
2. Dataset size and feature overlap still constrain separability.
3. Tuning budget can be expanded for deeper search.

### Practical next improvements

1. Add stronger feature engineering around opponent context and game context.
2. Explore calibrated probabilities and threshold-specific operating policies.
3. Try repeated time-based validation folds for more robust tuning estimates.
4. Evaluate focal loss or class-balanced alternatives.
5. Add lightweight ensembling between top NN trials.

---

## 15. Short Form Summary (Can become poster abstract)

We built a full supervised learning pipeline to predict whether LeBron James will record a high-scoring game (>=25 points). Among three required model families, the Neural Network achieved the strongest overall performance under our project selection criteria. The final model is a regularized MLP trained with BCEWithLogitsLoss, class weighting, AdamW, validation-guided LR scheduling, and early stopping. We also tuned the decision threshold on validation data and performed validation-first randomized hyperparameter search using focused and exploratory branches. Final test results reached 65.9% accuracy and 0.6051 macro-F1 at threshold 0.385, with the Neural Network selected as the best model for balanced predictive performance in this dataset.
