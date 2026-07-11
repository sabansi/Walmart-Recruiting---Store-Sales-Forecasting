"""
Shared WMAE (Weighted Mean Absolute Error) scorer — Walmart Sales Forecasting project.
Canonical version. Both team members import from here.

Official competition metric: holiday weeks weighted 5x, all other weeks weighted 1x.
Every model notebook must use this file — do not reimplement the metric locally,
or model comparisons across notebooks/architectures become invalid.
"""

import numpy as np
import pandas as pd


def wmae(y_true, y_pred, is_holiday):
    """
    Weighted Mean Absolute Error — official Walmart competition metric.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n,)
    is_holiday : array-like of bool/int, shape (n,)

    Returns
    -------
    float (lower is better)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.where(np.asarray(is_holiday, dtype=bool), 5.0, 1.0)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def wmae_from_df(df, pred_col="Weekly_Sales_pred", true_col="Weekly_Sales", holiday_col="IsHoliday"):
    """Convenience wrapper that accepts a DataFrame."""
    return wmae(df[true_col].values, df[pred_col].values, df[holiday_col].values)


def wmae_breakdown(y_true, y_pred, is_holiday, cluster_id=None):
    """
    Compute WMAE overall, split by holiday/non-holiday, and optionally per cluster.

    Every model notebook should report this breakdown, not just a single overall
    number — it's the basis for the segment-dependent narrative in the final
    report (which architecture wins on holiday weeks vs regular weeks, and on
    which store-dept clusters).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    is_holiday = np.asarray(is_holiday, dtype=bool)

    result = {
        "overall_wmae": wmae(y_true, y_pred, is_holiday),
        "holiday_wmae": wmae(y_true[is_holiday], y_pred[is_holiday], is_holiday[is_holiday])
            if is_holiday.sum() > 0 else np.nan,
        "non_holiday_wmae": wmae(y_true[~is_holiday], y_pred[~is_holiday], is_holiday[~is_holiday])
            if (~is_holiday).sum() > 0 else np.nan,
        "n_holiday_rows": int(is_holiday.sum()),
        "n_non_holiday_rows": int((~is_holiday).sum()),
    }

    if cluster_id is not None:
        cluster_id = np.asarray(cluster_id)
        df = pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred,
            "is_holiday": is_holiday, "cluster_id": cluster_id,
        })
        per_cluster = (
            df.groupby("cluster_id")
              .apply(lambda g: wmae(g["y_true"], g["y_pred"], g["is_holiday"]), include_groups=False)
              .rename("wmae")
              .reset_index()
        )
        result["per_cluster_wmae"] = per_cluster

    return result


def print_wmae_report(result, model_name=""):
    """Pretty-print a wmae_breakdown() result — call at the end of every CV/eval run."""
    print(f"===== WMAE report: {model_name} =====")
    print(f"Overall WMAE:      {result['overall_wmae']:.3f}")
    print(f"Holiday WMAE:      {result['holiday_wmae']:.3f}  (n={result['n_holiday_rows']})")
    print(f"Non-holiday WMAE:  {result['non_holiday_wmae']:.3f}  (n={result['n_non_holiday_rows']})")
    if "per_cluster_wmae" in result:
        print("\nPer-cluster WMAE:")
        print(result["per_cluster_wmae"].to_string(index=False))