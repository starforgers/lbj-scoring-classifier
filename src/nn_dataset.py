from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: Optional[np.ndarray]
    X_test: np.ndarray
    y_test: Optional[np.ndarray]
    feature_names: list[str]
    scaler: StandardScaler


def _normalize_date_column(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in frame.columns:
        return frame

    converted = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.copy()
    frame[date_col] = converted.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    return frame


def _split_xy(
    frame: pd.DataFrame,
    target_col: str,
    drop_cols: Iterable[str],
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    frame = frame.copy()
    y: Optional[pd.Series] = None

    if target_col in frame.columns:
        y = pd.to_numeric(frame[target_col], errors="coerce")
        frame = frame.drop(columns=[target_col])

    removable = [column for column in drop_cols if column in frame.columns]
    if removable:
        frame = frame.drop(columns=removable)

    X = pd.get_dummies(frame, drop_first=False)
    return X, y


def _as_float32_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    aligned = frame.reindex(columns=columns, fill_value=0.0)
    return aligned.astype(np.float32).to_numpy()


def _as_float32_vector(series: Optional[pd.Series]) -> Optional[np.ndarray]:
    if series is None:
        return None
    return series.astype(np.float32).to_numpy()


def load_tabular_splits(
    train_path: str,
    valid_path: str,
    test_path: str,
    target_col: str = "HSG",
    date_col: str = "Date",
    drop_cols: Optional[Iterable[str]] = None,
) -> TabularSplits:
    drop_cols = list(drop_cols or [])

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    train_df = _normalize_date_column(train_df, date_col)
    valid_df = _normalize_date_column(valid_df, date_col)
    test_df = _normalize_date_column(test_df, date_col)

    X_train_df, y_train_series = _split_xy(train_df, target_col=target_col, drop_cols=drop_cols)
    X_valid_df, y_valid_series = _split_xy(valid_df, target_col=target_col, drop_cols=drop_cols)
    X_test_df, y_test_series = _split_xy(test_df, target_col=target_col, drop_cols=drop_cols)

    feature_names = list(X_train_df.columns)

    X_train = _as_float32_matrix(X_train_df, feature_names)
    X_valid = _as_float32_matrix(X_valid_df, feature_names)
    X_test = _as_float32_matrix(X_test_df, feature_names)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_valid = scaler.transform(X_valid).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    y_train = _as_float32_vector(y_train_series)
    y_valid = _as_float32_vector(y_valid_series)
    y_test = _as_float32_vector(y_test_series)

    if y_train is None:
        raise ValueError(f"Target column '{target_col}' was not found in train split: {train_path}")

    return TabularSplits(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )
