import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from nn_dataLoader import DataLoader
from nn_model import NeuralNetwork


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    X_batches, y_batches = loader.get_batch(mode="train" if train_mode else "test")

    total_loss = 0.0
    n_samples = 0

    for X_np, y_np in zip(X_batches, y_batches):
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_np, dtype=torch.float32, device=device).view(-1, 1)

        if train_mode:
            optimizer.zero_grad()
            logits, _ = model(X_t)
            per_sample_loss = criterion(logits, y_t)
            loss = per_sample_loss.mean()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits, _ = model(X_t)
                per_sample_loss = criterion(logits, y_t)
                loss = per_sample_loss.mean()

        batch_n = y_t.size(0)
        total_loss += loss.item() * batch_n
        n_samples += batch_n

    return total_loss / max(n_samples, 1)


def predict_probabilities(model, X, y, batch_size=64, device="cpu"):
    loader = DataLoader(X, y, batch_size=batch_size)
    X_batches, _ = loader.get_batch(mode="test")

    all_probs = []
    model.eval()
    with torch.no_grad():
        for X_np in X_batches:
            X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
            _, probs = model(X_t)
            all_probs.append(probs.detach().cpu().numpy().reshape(-1))

    return np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)


def find_best_threshold_f1(y_true, y_prob, grid):
    best_thr = 0.5
    best_macro_f1 = -1.0
    best_bal_acc = -1.0
    best_acc = -1.0

    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, pred)
        acc = accuracy_score(y_true, pred)

        better = (
            (macro_f1 > best_macro_f1 + 1e-12)
            or (np.isclose(macro_f1, best_macro_f1) and bal_acc > best_bal_acc + 1e-12)
            or (
                np.isclose(macro_f1, best_macro_f1)
                and np.isclose(bal_acc, best_bal_acc)
                and acc > best_acc + 1e-12
            )
        )

        if better:
            best_thr = float(thr)
            best_macro_f1 = float(macro_f1)
            best_bal_acc = float(bal_acc)
            best_acc = float(acc)

    return best_thr, best_macro_f1, best_bal_acc, best_acc


def run_tuning_search(
    splits,
    max_trials,
    trial_max_epochs,
    trial_patience,
    trial_threshold_grid,
    pos_weight_bce,
    device,
    sample_cfg,
    rng=None,
    focused_prob=0.90,
    progress_every=5,
    verbose=True,
):
    if rng is None:
        rng = np.random.default_rng()

    random_search_results = []
    best_by_valid = None
    y_valid_local = splits.y_valid.astype(int)

    for trial in range(1, max_trials + 1):
        cfg = sample_cfg(rng, focused=(rng.random() < focused_prob))

        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])

        train_loader_local = DataLoader(splits.X_train, splits.y_train, batch_size=cfg["batch_size"])
        valid_loader_local = DataLoader(splits.X_valid, splits.y_valid, batch_size=cfg["batch_size"])

        model_local = NeuralNetwork(
            input_size=splits.X_train.shape[1],
            num_classes=1,
            list_hidden=cfg["hidden"],
            activation=cfg["activation"],
            dropout_p=cfg["dropout"],
        )
        model_local.create_network()
        model_local.init_weights(seed=cfg["seed"])
        model_local = model_local.to(device)

        pos_weight_tensor_local = torch.tensor([pos_weight_bce], dtype=torch.float32, device=device)
        criterion_local = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor_local)
        optimizer_local = torch.optim.AdamW(model_local.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        scheduler_local = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_local, mode="min", factor=0.5, patience=10, min_lr=5e-6
        )

        best_state_local = copy.deepcopy(model_local.state_dict())
        best_val_loss_local = float("inf")
        stale = 0

        for _ in range(trial_max_epochs):
            _ = run_epoch(
                model_local,
                train_loader_local,
                criterion_local,
                optimizer=optimizer_local,
                device=device,
            )
            valid_loss_epoch = run_epoch(
                model_local,
                valid_loader_local,
                criterion_local,
                optimizer=None,
                device=device,
            )
            scheduler_local.step(valid_loss_epoch)

            if valid_loss_epoch < best_val_loss_local - 1e-12:
                best_val_loss_local = valid_loss_epoch
                best_state_local = copy.deepcopy(model_local.state_dict())
                stale = 0
            else:
                stale += 1

            if stale >= trial_patience:
                break

        model_local.load_state_dict(best_state_local)

        valid_probs_local = predict_probabilities(
            model_local,
            splits.X_valid,
            splits.y_valid,
            batch_size=cfg["batch_size"],
            device=device,
        )
        best_thr_local, valid_f1_local, valid_bal_acc_local, valid_acc_local = find_best_threshold_f1(
            y_valid_local,
            valid_probs_local,
            trial_threshold_grid,
        )

        row = {
            "trial": trial,
            "seed": cfg["seed"],
            "hidden": str(cfg["hidden"]),
            "activation": cfg["activation"],
            "batch_size": cfg["batch_size"],
            "dropout": float(cfg["dropout"]),
            "lr": float(cfg["lr"]),
            "wd": float(cfg["wd"]),
            "threshold": float(best_thr_local),
            "valid_f1": float(valid_f1_local),
            "valid_bal_acc": float(valid_bal_acc_local),
            "valid_acc": float(valid_acc_local),
            "val_loss": float(best_val_loss_local),
            "model": copy.deepcopy(model_local.state_dict()),
        }
        random_search_results.append(row)

        if (best_by_valid is None) or (row["valid_f1"] > best_by_valid["valid_f1"]):
            best_by_valid = row

        if verbose and (trial == 1 or trial % progress_every == 0):
            print(
                f"Trial {trial:04d}/{max_trials} | best validation Macro-F1: {best_by_valid['valid_f1']:.4f}",
                flush=True,
            )

    results_df = pd.DataFrame(random_search_results)
    return results_df, best_by_valid
