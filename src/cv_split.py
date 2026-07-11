"""
Shared time-based train/validation split — Walmart Sales Forecasting project.
Canonical version. Both team members import from here.

Train.csv covers 2010-02-05 -> 2012-10-26 (143 weeks).
Test.csv (Kaggle, unlabeled) covers 2012-11-02 -> 2013-07-26 (39 weeks ahead).

VAL_WEEKS = 13 (one quarter) is the team-wide agreed validation window: it lines
up with the Sales_Lag13 / Sales_Roll_13w features in features.py, and 13 weeks
is a large enough holdout to include a real holiday week for a meaningful
holiday-vs-non-holiday WMAE breakdown.

Returns integer row positions (not sub-dataframes) so this plugs directly into
sklearn-style CV loops without repeatedly re-filtering a full dataframe.
"""

import numpy as np
import pandas as pd

VAL_WEEKS = 13  # <-- fixed team-wide constant. Do not change per-notebook.


def time_series_cv_splits(df, date_col="Date", n_splits=3, val_weeks=VAL_WEEKS):
    """
    Walk-forward time-series cross-validation splits.

    Validation windows are taken from the END of the training period, walking
    backwards by val_weeks each fold (fold 0 = most recent window). No future
    leakage: each fold's train_idx only contains rows strictly before that
    fold's validation window.

    Yields
    ------
    (train_idx, val_idx) : tuple of np.ndarray of integer row positions
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    dates = sorted(df[date_col].unique())
    n_dates = len(dates)

    for fold in range(n_splits):
        val_end_idx = n_dates - fold * val_weeks
        val_start_idx = val_end_idx - val_weeks

        if val_start_idx <= 0:
            break

        train_dates = set(dates[:val_start_idx])
        val_dates = set(dates[val_start_idx:val_end_idx])

        train_idx = np.where(df[date_col].isin(train_dates).values)[0]
        val_idx = np.where(df[date_col].isin(val_dates).values)[0]

        yield train_idx, val_idx


def last_cutoff_split(df, date_col="Date", val_weeks=VAL_WEEKS):
    """Single train/val split using the most recent val_weeks as validation.
    This is the default split every model notebook should use for its first-pass CV run."""
    splits = list(time_series_cv_splits(df, date_col=date_col, n_splits=1, val_weeks=val_weeks))
    return splits[0]


def time_based_split(df, date_col="Date", n_val_weeks=VAL_WEEKS):
    """
    Convenience wrapper returning actual train/val DataFrames (rather than index
    positions) for notebooks that prefer working directly with dataframes.
    """
    train_idx, val_idx = last_cutoff_split(df, date_col=date_col, val_weeks=n_val_weeks)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df