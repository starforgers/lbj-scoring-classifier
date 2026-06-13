# Neural Network Hyperparameter Tuning Explanation (Step 9)

This document explains the tuning variables, ranges, and logic used in the validation-first randomized search in Step 9.

## 1) Tuning Objective

- Primary objective: maximize validation Macro-F1.
- Secondary tracked metrics: validation balanced accuracy, validation accuracy, and validation loss.
- Test metrics are not used to select the best configuration.

## 2) Search Budget and Control Variables

- max_trials = 200
- trial_max_epochs = 320
- trial_patience = 40
- progress_every = 50
- target_valid_f1 = 0.57 (optional early stop target)

Meaning:

- `max_trials`: maximum number of sampled configurations.
- `trial_max_epochs`: maximum training epochs per trial.
- `trial_patience`: stops a trial if validation loss does not improve for 40 epochs.
- `progress_every`: logging frequency for long runs.
- `target_valid_f1`: search can stop early when target quality is reached.

## 3) Variables Being Tuned

- hidden (network architecture)
- activation
- batch_size
- dropout
- lr (learning rate)
- wd (weight decay)
- seed
- threshold (classification threshold tuned per trial)

## 4) Candidate Sets and Global Ranges

- hidden_candidates:
  - [128, 64]
  - [192, 96]
  - [256, 128]
  - [320, 128]
  - [384, 192]
  - [448, 224]
  - [256, 128, 64]
  - [384, 192, 96]
- activation_candidates: ["relu", "tanh"]
- batch_size_candidates: [16, 32, 64]
- seed_candidates: [13, 21, 34, 42, 55, 67, 78, 89, 99, 123, 144, 233, 377, 610, 777, 987]
- trial_threshold_grid: np.linspace(0.10, 0.90, 161)

Threshold notes:

- 161 points from 0.10 to 0.90 gives a step of 0.005.
- Best threshold is selected per trial using Macro-F1, with tie-breaks.

## 5) Sampling Strategy: Focused vs Unfocused

Each trial samples a config using:

- Focused mode (90% probability): sample from hand-crafted promising branches.
- Unfocused mode (10% probability): broad exploration to preserve diversity.

This balances exploitation (good known regions) and exploration (new regions).

## 6) Focused Branches and Ranges

### Branch 0 (compact-to-medium ReLU, light regularization)

- hidden: indices [1, 2, 3] -> [192, 96], [256, 128], [320, 128]
- activation: relu
- batch_size: [32, 16]
- dropout: uniform [0.00, 0.10]
- lr: log-uniform [2.0e-4, 8.5e-4]
- wd: log-uniform [1e-6, 4e-5]

### Branch 1 (wider ReLU, stronger regularization)

- hidden: indices [4, 3, 2] -> [384, 192], [320, 128], [256, 128]
- activation: relu
- batch_size: [16, 32]
- dropout: uniform [0.02, 0.18]
- lr: log-uniform [1.5e-4, 7.5e-4]
- wd: log-uniform [5e-6, 1.5e-4]

### Branch 2 (tanh branch, higher-capacity range)

- hidden: indices [4, 5, 7] -> [384, 192], [448, 224], [384, 192, 96]
- activation: tanh
- batch_size: [16, 32]
- dropout: uniform [0.10, 0.30]
- lr: log-uniform [3e-4, 1.8e-3]
- wd: log-uniform [2e-5, 3e-4]

### Branch 3 fallback (conservative exploit region)

- seed: [610, 55, 89, 13, 99, 123, 233, 377]
- hidden: indices [1, 2] -> [192, 96], [256, 128]
- activation: relu
- batch_size: [32, 16]
- dropout: uniform [0.00, 0.035]
- lr: log-uniform [2.4e-4, 4.8e-4]
- wd: log-uniform [4e-6, 5e-5]

## 7) Unfocused Mode (Full-Space Exploration)

- hidden: sampled from all hidden_candidates
- activation: sampled from ["relu", "tanh"]
- batch_size: sampled from [16, 32, 64]
- dropout: uniform [0.00, 0.40]
- lr: log-uniform [8e-5, 2.5e-3]
- wd: log-uniform [1e-6, 4e-4]

## 8) Why lr and wd Use Log-Uniform Sampling

- Useful learning rates and weight decay values often span multiple orders of magnitude.
- Log-uniform avoids over-sampling only large values and gives fair coverage of small values.

## 9) Training and Trial Evaluation Logic

Per trial:

1. Sample config (`sample_cfg`).
2. Reseed random, numpy, and torch for reproducibility.
3. Build DataLoader objects using sampled batch size.
4. Build model with sampled hidden, activation, dropout.
5. Train with AdamW using sampled lr and wd.
6. Use ReduceLROnPlateau scheduler:
   - factor = 0.5
   - patience = 10
   - min_lr = 5e-6
7. Track best checkpoint by validation loss.
8. Stop early when no validation-loss improvement for `trial_patience` epochs.
9. Restore best checkpoint.
10. Predict validation probabilities and tune threshold over `trial_threshold_grid`.
11. Record trial row with all settings + validation metrics.

## 10) Threshold Selection Rule

`find_best_threshold_f1` selects threshold by:

1. highest Macro-F1,
2. if tied, higher balanced accuracy,
3. if still tied, higher accuracy.

## 11) Ranking and Final Selection

- Every trial is stored in `random_search_results`.
- Trials are ranked into `results_top_valid` by `valid_f1` descending.
- Best trial for tuning is the one with top validation Macro-F1.

## 12) Safeguard for Section 8 Comparison

Step 9 saves and restores Section 8 baseline comparison variables:

- nn_test_acc
- nn_test_f1
- y_true_nn
- test_preds_nn

This prevents Step 9 experimentation from overwriting earlier comparison outputs.

## 13) Practical Interpretation of Ranges

- Lower dropout/wd ranges (Branch 0 and fallback) favor less-regularized, stable learning.
- Higher dropout/wd ranges (Branch 1 and especially Branch 2) combat overfitting for larger models.
- Wider lr ranges in Branch 2 allow more aggressive optimization for tanh/high-capacity settings.
- Unfocused ranges are intentionally broad to discover useful regions not covered by focused branches.

## 14) Why We Chose Wider ReLU + Higher Regularization (Branch Design Rationale)

Short version:

- We expected wider ReLU networks to capture richer nonlinear feature interactions.
- We also expected wider models to overfit on a limited tabular dataset.
- So we paired higher capacity with stronger regularization (dropout and weight decay).

Detailed rationale:

1. Start from baseline behavior

- Earlier neural-network runs showed ReLU as a stable and competitive activation for this task.
- Because of that, multiple focused branches were allocated to ReLU regions.

2. Capacity-versus-overfitting tradeoff

- Wider/deeper hidden stacks increase representational capacity.
- On finite data, higher capacity can improve fit but also increase variance.
- Branch 1 was designed to test this tradeoff directly: wider ReLU plus stronger regularization.

3. Role of stronger regularization in wider branches

- Higher dropout reduces co-adaptation and improves robustness.
- Higher weight decay discourages overly large weights and helps generalization.
- Together, they make high-capacity models less likely to memorize noise.

4. Why branch structure instead of one flat range

- A single broad range can waste many trials in weak regions.
- Branching lets us encode targeted hypotheses:
  - Branch 0: medium ReLU, lighter regularization (stable baseline exploit).
  - Branch 1: wider ReLU, stronger regularization (high-capacity but controlled).
  - Branch 2: tanh high-capacity alternative with broader optimization ranges.
  - Branch 3: conservative fallback around prior-good settings.

5. Focused plus unfocused sampling

- 90% focused sampling improves search efficiency in promising zones.
- 10% unfocused sampling preserves global exploration and reduces local optimum risk.

6. Final selection remains data-driven

- Branches provide search priors, not hard assumptions.
- The winner is still determined by validation Macro-F1 (with tie-breaks), after per-trial early stopping and threshold tuning.

In summary, wider ReLU with stronger regularization was chosen as a deliberate, testable hypothesis: gain nonlinear expressiveness while actively controlling overfitting.

## 15) Full Neural Network Flowchart and Process Notes (End-to-End)

This section summarizes the complete neural network workflow implemented in the notebook, from data loading to model comparison handoff.

### 15.1 Flowchart (Notebook Process)

```mermaid
flowchart TD
    A[Step 1: Set seed and load train/valid/test splits] --> B[Step 2: Set training hyperparameters and class weights]
    B --> C[Step 3: Build DataLoader objects]
    C --> D[Step 4: Initialize NeuralNetwork model]
    D --> E[Step 5: Define run_epoch helper]
    E --> F[Step 6: Baseline training with early stopping on validation loss]
    F --> G[Step 7: Plot train vs validation loss]
    G --> H[Step 8: Baseline evaluation and threshold tuning on validation]
    H --> I[Step 8A: Baseline confusion matrix on test]
    I --> J[Step 9: Randomized hyperparameter search (validation-first)]
    J --> K[Step 9A-9B: Rank and inspect validation trials]
    K --> L[Restore Section 8 comparison variables]
    L --> M[Section 8: Compare LR vs KNN vs NN and select best model]
```

### 15.2 Step-by-Step Notes

1. Data and reproducibility setup

- The workflow starts by fixing random seeds for `random`, `numpy`, and `torch`.
- Data is loaded through `load_tabular_splits(...)` into train, validation, and test arrays.
- This ensures deterministic behavior and consistent split usage across all NN experiments.

2. Core training configuration

- Baseline defaults are set (batch size, learning rate, weight decay, max epochs, patience).
- Threshold grid is prepared: `np.linspace(0.10, 0.90, 161)`.
- Class imbalance is handled by computing `pos_weight_bce = neg_count / pos_count`.
- Device is selected automatically (`cuda` if available, else `cpu`).

3. Mini-batch data pipeline

- Custom `DataLoader` objects are created for train/validation/test splits.
- Batch size can change during tuning trials, so trial-specific loaders are rebuilt in Step 9.

4. Model architecture and optimizer stack

- Model class: `NeuralNetwork` from `nn_model.py`.
- Baseline architecture in notebook: hidden layers `[192, 96]`, activation `relu`, dropout around `0.058`.
- Loss: `BCEWithLogitsLoss` with `pos_weight` for class imbalance.
- Optimizer: `AdamW`.
- Scheduler: `ReduceLROnPlateau` to lower LR when validation loss stalls.

5. Epoch execution abstraction

- `run_epoch(...)` encapsulates:
  - train mode vs eval mode,
  - forward pass,
  - per-sample BCE loss,
  - backward pass and optimizer step (train mode only),
  - averaged epoch loss.
- This avoids duplicated training/evaluation code and keeps behavior consistent.

6. Baseline training loop (loss-guided)

- For each epoch:
  - train loss is computed,
  - validation loss is computed,
  - scheduler steps on validation loss,
  - best checkpoint is updated when validation loss improves.
- Early stopping triggers when validation loss fails to improve for `patience` epochs.
- Best checkpoint is restored before evaluation.

7. Baseline evaluation protocol

- Probabilities are produced with `predict_probabilities(...)`.
- Validation threshold is tuned by `find_best_threshold_f1(...)`:
  1. maximize Macro-F1,
  2. tie-break by balanced accuracy,
  3. then by accuracy.
- Selected threshold is applied to test probabilities for baseline test metrics.
- Test confusion matrix is plotted for class-wise error interpretation.

8. Validation-first hyperparameter search (Step 9)

- Randomized search explores architecture/optimization/training settings.
- Sampling is mostly focused (90%) and partly broad (10%).
- Each trial performs:
  - config sampling,
  - reseeding,
  - trial-specific loader/model/optimizer build,
  - training with early stopping by validation loss,
  - threshold tuning on validation,
  - metric logging and ranking record.
- Best trial is selected by validation Macro-F1 only.

9. Ranking, diagnostics, and safeguards

- Trial outputs are ranked into `results_top_valid` for inspection.
- Ranking plot and top-trials table are displayed in Step 9A and Step 9B.
- Section 8 baseline comparison variables are restored after Step 9 to prevent accidental overwrite.

10. Final project handoff to model comparison

- Section 8 compares Logistic Regression, KNN, and Neural Network on the same test split.
- Selection rule uses F1 first, then accuracy, then balanced accuracy.
- NN outputs (`y_true_nn`, `test_preds_nn`, `nn_test_f1`, `nn_test_acc`) are used for fair side-by-side comparison.

### 15.3 Why This Process Design Is Strong

- Uses a clear validation-first protocol (reduces test leakage risk).
- Separates threshold tuning from weight fitting.
- Preserves reproducibility via consistent seeding.
- Combines early stopping, dropout, weight decay, and LR scheduling for stable generalization.
- Maintains Section 8 comparability by restoring baseline handoff variables.

### 15.4 Quick Recap Notes

- Objective: maximize validation Macro-F1, not test metrics.
- Model family: custom MLP (`NeuralNetwork`) with configurable hidden layers.
- Baseline architecture used in notebook: two hidden layers (`[192, 96]`).
- Step 9 broadens search over architecture and optimization while keeping selection validation-driven.
- Final winner for the project is decided in Section 8 with a unified rule across all three models.
