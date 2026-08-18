"""Dataset loading and validation.

Kept separate from the modelling code so the assignment's dataset constraints are
checked in exactly one place, and so a different CSV can be swapped in without
touching anything else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_FEATURES = 12
MIN_INSTANCES = 500

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(os.path.dirname(HERE), "data", "wdbc.csv")
DEFAULT_TARGET = "diagnosis"


@dataclass(frozen=True)
class Dataset:
    """A loaded, validated dataset plus the facts the rest of the code needs."""

    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    target_col: str
    class_names: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    def summary(self) -> dict:
        counts = self.target.value_counts().to_dict()
        return {
            "instances": int(len(self.frame)),
            "features": int(self.features.shape[1]),
            "classes": self.n_classes,
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "numeric": len(self.numeric_cols),
            "categorical": len(self.categorical_cols),
            "missing_cells": int(self.frame.isna().sum().sum()),
        }


class DatasetError(ValueError):
    """Raised when a CSV does not satisfy the assignment's constraints."""


def load(csv_path: str = DEFAULT_CSV,
         target_col: str = DEFAULT_TARGET,
         drop_cols: list[str] | None = None) -> Dataset:
    """Read a CSV and return a validated Dataset.

    Raises DatasetError with a specific message rather than letting a vague
    KeyError surface three call frames later.
    """
    if not os.path.exists(csv_path):
        raise DatasetError(f"No such file: {csv_path}")

    frame = pd.read_csv(csv_path)
    if drop_cols:
        frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns])

    if target_col not in frame.columns:
        raise DatasetError(
            f"Target column {target_col!r} not found. "
            f"Available columns: {list(frame.columns)}"
        )

    features = frame.drop(columns=[target_col])
    target = frame[target_col]

    if features.shape[1] < MIN_FEATURES:
        raise DatasetError(
            f"The assignment requires at least {MIN_FEATURES} features; "
            f"this file has {features.shape[1]}."
        )
    if len(frame) < MIN_INSTANCES:
        raise DatasetError(
            f"The assignment requires at least {MIN_INSTANCES} instances; "
            f"this file has {len(frame)}."
        )
    if target.nunique() < 2:
        raise DatasetError("The target column has fewer than two distinct values.")

    numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]
    class_names = sorted(str(c) for c in target.unique())

    return Dataset(
        frame=frame,
        features=features,
        target=target.astype(str),
        target_col=target_col,
        class_names=class_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
