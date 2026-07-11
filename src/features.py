"""
Shared feature-engineering module — Walmart Sales Forecasting project.
Canonical version. Both team members import from here; do not fork it.

Merges the best of both drafts:
- Explicit feature-list constants (BASE_FEATURES / LAG_FEATURES / ALL_FEATURES / CATBOOST_CAT_FEATURES)
- Holiday proximity features (WeeksBefore_X / WeeksAfter_X)
- min_periods rolling stats (keeps more early-history rows instead of dropping them to NaN)
- Zip-aware raw data loading

Fixes vs the earlier draft:
- build_lag_features no longer uses groupby().transform(lambda ...): that pattern loops
  in Python per group (3,331 groups) and took ~17s on this dataset. Replaced with a
  vectorized groupby-shift-rolling chain (~2s).
- Added add_origin_style_features(): the test set requires predicting 39 weeks straight
  ahead in one shot. Only Sales_Lag52 is knowable across that entire horizon; every
  shorter lag (1/2/4/13/26) is only real for the first few test weeks and NaN after
  that — it's not missing data to impute, it's structurally unavailable at forecast
  time. This function computes: (a) a correct row-by-row Sales_Lag52, and (b) a set of
  "as of forecast origin" rolling stats computed ONCE from history up to a cutoff date
  and broadcast identically across every row being forecast, matching what a real
  direct-multi-horizon model actually has available.
  IMPORTANT: use this same function (not build_lag_features) when scoring your
  validation split too — otherwise your local WMAE will look better than your real
  Kaggle leaderboard WMAE, because validation would be "cheating" with per-row lags
  that the real test set doesn't have access to.
"""

import numpy as np
import pandas as pd
import zipfile
import os
from functools import reduce


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _maybe_unzip(zip_path, extract_dir):
    """Unzip a file if the extracted CSV does not already exist."""
    csv_name = os.path.basename(zip_path).replace(".zip", "")
    csv_path = os.path.join(extract_dir, csv_name)
    if not os.path.exists(csv_path) and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
    return csv_path if os.path.exists(csv_path) else os.path.join(extract_dir, csv_name)


def load_raw_data(data_dir="."):
    """
    Load train, test, features, and stores CSVs (unzipping if a .zip is present
    and the .csv hasn't been extracted yet).
    """
    train_path  = _maybe_unzip(os.path.join(data_dir, "train.csv.zip"), data_dir) \
        if os.path.exists(os.path.join(data_dir, "train.csv.zip")) else os.path.join(data_dir, "train.csv")
    test_path   = _maybe_unzip(os.path.join(data_dir, "test.csv.zip"), data_dir) \
        if os.path.exists(os.path.join(data_dir, "test.csv.zip")) else os.path.join(data_dir, "test.csv")
    feat_path   = _maybe_unzip(os.path.join(data_dir, "features.csv.zip"), data_dir) \
        if os.path.exists(os.path.join(data_dir, "features.csv.zip")) else os.path.join(data_dir, "features.csv")
    stores_path = os.path.join(data_dir, "stores.csv")

    train    = pd.read_csv(train_path,  parse_dates=["Date"])
    test     = pd.read_csv(test_path,   parse_dates=["Date"])
    features = pd.read_csv(feat_path,   parse_dates=["Date"])
    stores   = pd.read_csv(stores_path)
    return train, test, features, stores


def merge_all(train, test, features, stores):
    """Left-join features and stores onto train and test."""
    train = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    train = train.merge(stores,   on="Store",                        how="left")
    test  = test.merge(features,  on=["Store", "Date", "IsHoliday"], how="left")
    test  = test.merge(stores,    on="Store",                        how="left")
    return train, test


# ---------------------------------------------------------------------------
# Holiday calendars
# ---------------------------------------------------------------------------

SUPER_BOWL_DATES   = pd.to_datetime(["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"])
LABOR_DAY_DATES    = pd.to_datetime(["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"])
THANKSGIVING_DATES = pd.to_datetime(["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"])
CHRISTMAS_DATES    = pd.to_datetime(["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"])


def build_calendar_features(df):
    """Add year/month/week/quarter, per-holiday binary flags, and proximity-to-holiday features."""
    df = df.copy()
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]    = df["Date"].dt.quarter

    df["IsSuperBowl"]    = df["Date"].isin(SUPER_BOWL_DATES).astype(int)
    df["IsLaborDay"]     = df["Date"].isin(LABOR_DAY_DATES).astype(int)
    df["IsThanksgiving"] = df["Date"].isin(THANKSGIVING_DATES).astype(int)
    df["IsChristmas"]    = df["Date"].isin(CHRISTMAS_DATES).astype(int)

    for hol_name, hol_dates in [("SuperBowl", SUPER_BOWL_DATES),
                                 ("Thanksgiving", THANKSGIVING_DATES),
                                 ("Christmas", CHRISTMAS_DATES)]:
        before_col = f"WeeksBefore_{hol_name}"
        after_col  = f"WeeksAfter_{hol_name}"
        df[before_col] = 0
        df[after_col]  = 0
        for h in hol_dates:
            m_before = (df["Date"] >= h - pd.Timedelta(weeks=3)) & (df["Date"] < h)
            m_after  = (df["Date"] > h) & (df["Date"] <= h + pd.Timedelta(weeks=2))
            df.loc[m_before, before_col] = ((h - df.loc[m_before, "Date"]).dt.days // 7).clip(0, 3)
            df.loc[m_after,  after_col]  = ((df.loc[m_after,  "Date"] - h).dt.days // 7).clip(0, 2)

    return df


def build_markdown_features(df):
    """Clean and aggregate the 5 markdown columns."""
    df = df.copy()
    for col in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).clip(lower=0.0)
        else:
            df[col] = 0.0
    df["MarkDown_Total"] = df[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]].sum(axis=1)
    df["MarkDown_Count"] = (df[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]] > 0).sum(axis=1)
    return df


def fill_macro_features(df):
    """
    CPI/Unemployment are missing only for the last few weeks of features.csv
    (not yet published at extraction time). Forward/back-fill per store since
    these are slow-moving macro series.
    """
    df = df.copy().sort_values(["Store", "Date"])
    df[["CPI", "Unemployment"]] = df.groupby("Store")[["CPI", "Unemployment"]].ffill().bfill()
    return df


def build_store_features(df):
    """Encode store Type (A/B/C) as an integer for models that need numeric input.
    Tree models (LightGBM/CatBoost) should be told this is categorical, not ordinal —
    see CATBOOST_CAT_FEATURES below."""
    df = df.copy()
    if "Type" in df.columns:
        df["Type_Enc"] = df["Type"].map({"A": 0, "B": 1, "C": 2}).astype(int)
    return df


def build_lag_features(df):
    """
    Add lag and rolling-window features using real per-row history.

    Use this ONLY for a dataset where Weekly_Sales is known for every row up to
    and including that row's own recent past — i.e. the full training set, or
    a training slice ending before your validation cutoff.

    Do NOT call this on the real Kaggle test set, and do NOT call this
    naively on a validation split if you also computed lags on the full
    training set first (that would leak post-cutoff info into pre-cutoff
    rows via the shift). Call it on train and val SEPARATELY after your
    time_based_split, using only each split's own rows in the groupby.

    For test / true out-of-sample scoring, use add_origin_style_features() instead.
    """
    df = df.copy().sort_values(["Store", "Dept", "Date"])
    g = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    for lag in [1, 2, 4, 13, 26, 52]:
        df[f"Sales_Lag{lag}"] = g.shift(lag)

    # Vectorized rolling stats (fixed: no groupby().transform(lambda...), which
    # loops in pure Python per group and is ~8x slower on this dataset).
    shifted = g.shift(1)
    grouped_shifted = shifted.groupby([df["Store"], df["Dept"]])
    for window, alias in [(4, "4w"), (13, "13w"), (26, "26w"), (52, "52w")]:
        df[f"Sales_Roll_{alias}_Mean"] = (
            grouped_shifted.rolling(window, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        )
        if window <= 13:
            df[f"Sales_Roll_{alias}_Std"] = (
                grouped_shifted.rolling(window, min_periods=2).std().reset_index(level=[0, 1], drop=True)
            )
    return df


def add_origin_style_features(target_df, history_df, origin_date=None):
    """
    Add lag/rolling features to `target_df` (validation OR real test rows) using
    ONLY information available as of `origin_date` (defaults to history_df's max
    date) — i.e. what a direct multi-horizon forecaster genuinely has access to.

    - Sales_Lag52 is computed correctly per row: history reaches back far enough
      to cover it across the whole horizon (52 weeks from any target date falls
      inside history_df for both the validation and true-test use cases here).
    - Rolling mean/std over the last 4/13/26/52 weeks of history_df are computed
      ONCE per Store-Dept as of origin_date, then broadcast identically to every
      row of target_df for that Store-Dept (a real forecaster doesn't get fresh
      sales numbers to update these mid-horizon).

    Column names get an "_Origin" suffix so they're never confused with the
    real per-row build_lag_features() columns.

    Use this identically for:
      - Validation scoring: history_df = train rows strictly before the CV cutoff,
        target_df = validation rows, origin_date = the cutoff date.
      - Real Kaggle test scoring: history_df = full train.csv,
        target_df = test.csv rows, origin_date = train's max date.
    Using the SAME function for both keeps your local validation WMAE honest —
    it will no longer look better than your real leaderboard WMAE just because
    validation secretly had access to short lags that the real test set doesn't.
    """
    history_df = history_df.sort_values(["Store", "Dept", "Date"])
    target_df = target_df.copy()
    origin_date = origin_date or history_df["Date"].max()
    origin_date = pd.Timestamp(origin_date)

    target_df["Forecast_Horizon"] = (
        (pd.to_datetime(target_df["Date"]) - origin_date).dt.days // 7
    ).astype(int)

    # --- Sales_Lag52: real row-by-row value, valid across the whole horizon ---
    combined = pd.concat([
        history_df[["Store", "Dept", "Date", "Weekly_Sales"]],
        target_df[["Store", "Dept", "Date"]].assign(Weekly_Sales=np.nan),
    ], ignore_index=True).drop_duplicates(subset=["Store", "Dept", "Date"])
    combined = combined.sort_values(["Store", "Dept", "Date"])
    combined["Sales_Lag52_Origin"] = combined.groupby(["Store", "Dept"])["Weekly_Sales"].shift(52)
    target_df = target_df.merge(
        combined[["Store", "Dept", "Date", "Sales_Lag52_Origin"]],
        on=["Store", "Dept", "Date"], how="left"
    )

    # --- static rolling stats as of origin_date, broadcast to every target row ---
    frames = []
    for window, alias in [(4, "4w"), (13, "13w"), (26, "26w"), (52, "52w")]:
        recent = history_df[
            (history_df["Date"] > origin_date - pd.Timedelta(weeks=window)) &
            (history_df["Date"] <= origin_date)
        ]
        stat = recent.groupby(["Store", "Dept"])["Weekly_Sales"].agg(["mean", "std"]).reset_index()
        stat = stat.rename(columns={
            "mean": f"Sales_Roll_{alias}_Mean_Origin",
            "std": f"Sales_Roll_{alias}_Std_Origin",
        })
        frames.append(stat)

    origin_stats = reduce(lambda l, r: l.merge(r, on=["Store", "Dept"], how="outer"), frames)
    target_df = target_df.merge(origin_stats, on=["Store", "Dept"], how="left")
    return target_df


def build_origin_training_frame(
    df,
    horizon_weeks=39,
    origin_stride_weeks=13,
    min_history_weeks=52,
    origin_dates=None,
    require_full_horizon=True,
):
    """
    Build supervised CatBoost rows that match real forecast-time feature availability.

    Each candidate origin date uses only rows at or before that origin as history,
    then creates labeled examples for the next `horizon_weeks` known target weeks.
    The resulting rows use the same origin-style lag/rolling columns produced for
    validation and Kaggle test by add_origin_style_features().

    This intentionally avoids training on per-row rolling means that update inside
    the forecast horizon, because those updates are unavailable for the Kaggle test.
    """
    df = df.copy().sort_values(["Store", "Dept", "Date"])
    unique_dates = pd.Index(sorted(pd.to_datetime(df["Date"].unique())))

    if origin_dates is None:
        max_origin_pos = len(unique_dates) - (horizon_weeks if require_full_horizon else 1)
        if max_origin_pos <= min_history_weeks - 1:
            return pd.DataFrame(columns=df.columns)
        candidate_dates = unique_dates[min_history_weeks - 1:max_origin_pos]
        candidate_dates = candidate_dates[::max(1, int(origin_stride_weeks))]
    else:
        candidate_dates = pd.Index(pd.to_datetime(origin_dates))

    frames = []
    for origin_date in candidate_dates:
        history = df[df["Date"] <= origin_date].copy()
        target_end = origin_date + pd.Timedelta(weeks=horizon_weeks)
        target = df[(df["Date"] > origin_date) & (df["Date"] <= target_end)].copy()

        if target.empty:
            continue
        if require_full_horizon and target["Date"].nunique() < horizon_weeks:
            continue

        target_base = build_all_features(target, add_lags=False)
        target_fe = add_origin_style_features(target_base, history, origin_date=origin_date)
        target_fe["Forecast_Origin_Date"] = origin_date
        frames.append(target_fe)

    if not frames:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(frames, ignore_index=True)


def build_all_features(df, add_lags=True):
    """
    Full feature pipeline: calendar + markdown + macro fill + store + (optional) lags.
    Pass add_lags=False when you plan to call add_origin_style_features() separately
    (i.e. for validation or the real test set).
    """
    df = build_calendar_features(df)
    df = build_markdown_features(df)
    df = fill_macro_features(df)
    df = build_store_features(df)
    if add_lags:
        df = build_lag_features(df)
    return df


# ---------------------------------------------------------------------------
# Feature column lists — import these directly instead of hardcoding column names
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "Store", "Dept",
    "Year", "Month", "WeekOfYear", "Quarter",
    "IsSuperBowl", "IsLaborDay", "IsThanksgiving", "IsChristmas",
    "WeeksBefore_SuperBowl", "WeeksBefore_Thanksgiving", "WeeksBefore_Christmas",
    "WeeksAfter_SuperBowl", "WeeksAfter_Thanksgiving", "WeeksAfter_Christmas",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "MarkDown_Total", "MarkDown_Count",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Size", "Type_Enc",
    "IsHoliday",
]

# Use for a train/val split where build_lag_features() was called on each split separately
LAG_FEATURES_TRAIN = [
    "Sales_Lag1", "Sales_Lag2", "Sales_Lag4", "Sales_Lag13", "Sales_Lag26", "Sales_Lag52",
    "Sales_Roll_4w_Mean", "Sales_Roll_4w_Std",
    "Sales_Roll_13w_Mean", "Sales_Roll_13w_Std",
    "Sales_Roll_26w_Mean",
    "Sales_Roll_52w_Mean",
]

# Use for validation (scored honestly) or the real Kaggle test set, via add_origin_style_features()
LAG_FEATURES_ORIGIN = [
    "Forecast_Horizon",
    "Sales_Lag52_Origin",
    "Sales_Roll_4w_Mean_Origin", "Sales_Roll_4w_Std_Origin",
    "Sales_Roll_13w_Mean_Origin", "Sales_Roll_13w_Std_Origin",
    "Sales_Roll_26w_Mean_Origin",
    "Sales_Roll_52w_Mean_Origin",
]

ALL_FEATURES_TRAIN = BASE_FEATURES + LAG_FEATURES_TRAIN
ALL_FEATURES_ORIGIN = BASE_FEATURES + LAG_FEATURES_ORIGIN

CATBOOST_CAT_FEATURES = [
    "Store", "Dept", "Type_Enc", "Month", "WeekOfYear",
    "IsSuperBowl", "IsLaborDay", "IsThanksgiving", "IsChristmas", "IsHoliday",
]

TARGET = "Weekly_Sales"
