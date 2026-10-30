#!/usr/bin/env python3
"""
Electricity Bill Predictor
==========================
Forecasts DAILY household electricity consumption (kWh) from historical usage
and converts the forecast into an estimated electricity bill.

Dataset : UCI "Individual Household Electric Power Consumption"
Models  : Linear Regression  vs  Decision Tree Regressor  (+ naive baseline)
Metric  : Mean Absolute Error (MAE)  (RMSE and R2 are reported as extras)

Pipeline
--------
 1. Load the raw .txt file and convert it to CSV ('?' -> NaN)
 2. Clean the data (missing values) and aggregate minute data to daily kWh
 3. Explore the data (trend / monthly / weekday patterns)
 4. Feature engineering: date features + lag features + rolling mean
 5. Chronological train/test split (NO shuffling - it is a time series)
 6. Train Linear Regression and Decision Tree, evaluate with MAE
 7. Recursive multi-day forecast into the future
 8. Estimate the electricity bill with a slab tariff
 9. Save plots, metrics and forecast to the output folder

Usage
-----
    python main.py                          # auto-downloads dataset if missing
    python main.py --data path/to/household_power_consumption.txt
    python main.py --synthetic              # offline demo with fake data
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
RAW_TXT_NAME = "household_power_consumption.txt"
CSV_NAME = "household_power_consumption.csv"

LAGS = [1, 2, 3, 7, 14]   # "usage N days ago" features
ROLL_WINDOW = 7           # mean usage of the previous 7 days
MIN_DAY_COVERAGE = 0.5    # a day needs >= 50% valid minute readings, else NaN
RANDOM_STATE = 42

# Illustrative slab tariff (editable). Applied per 30-day billing period.
# (upper limit of slab in kWh, price per kWh). Slabs are cumulative.
CURRENCY = "₹"
TARIFF_SLABS = [(100, 3.5), (300, 5.5), (float("inf"), 7.5)]
FIXED_CHARGE = 50.0       # fixed charge per billing period
BILLING_DAYS = 30

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# 1. Data loading  (text -> CSV)
# --------------------------------------------------------------------------- #
def download_dataset(data_dir: Path) -> Path:
    """Download and unzip the UCI dataset into data_dir. Returns the .txt path."""
    print(f"Downloading dataset from UCI ...\n  {UCI_URL}")
    with urllib.request.urlopen(UCI_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir)
    return data_dir / RAW_TXT_NAME


def convert_txt_to_csv(txt_path: Path, csv_path: Path) -> pd.DataFrame:
    """
    Read the semicolon-separated raw file and save it as a proper CSV.
    Missing values are written as '?' in the raw file, so we map them to NaN.
    """
    print(f"Reading raw file: {txt_path}")
    df = pd.read_csv(
        txt_path,
        sep=";",
        na_values=["?", ""],       # '?' marks a missing reading
        low_memory=False,
    )
    # Combine the Date (dd/mm/yyyy) and Time (hh:mm:ss) columns into one datetime
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"])
    # Make sure every measurement column is numeric
    num_cols = [c for c in df.columns if c != "datetime"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df[["datetime"] + num_cols]

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV ({len(df):,} rows): {csv_path}")
    return df


def load_raw_data(data_arg: str | None, data_dir: Path) -> pd.DataFrame:
    """Return the minute-level dataframe, converting/downloading when needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / CSV_NAME

    if data_arg:                                   # user supplied a file
        p = Path(data_arg)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, parse_dates=["datetime"], na_values=["?", ""])
        return convert_txt_to_csv(p, csv_path)

    if csv_path.exists():                          # already converted earlier
        print(f"Using existing CSV: {csv_path}")
        return pd.read_csv(csv_path, parse_dates=["datetime"])

    txt_path = data_dir / RAW_TXT_NAME
    if not txt_path.exists():
        txt_path = download_dataset(data_dir)
    return convert_txt_to_csv(txt_path, csv_path)


# --------------------------------------------------------------------------- #
# 2. Cleaning + daily aggregation
# --------------------------------------------------------------------------- #
def clean_and_aggregate(raw: pd.DataFrame) -> pd.Series:
    """
    Minute-level data -> clean daily consumption in kWh.

    Global_active_power is in kilowatts (kW), averaged each minute.
    Daily energy (kWh) = mean power in kW over the day * 24 hours.
    Using the mean (instead of a plain sum) keeps days with a few missing
    minutes comparable to complete days.
    """
    print("\n=== CLEANING ===")
    df = raw.copy()
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    df = df.set_index("datetime")

    total_missing = int(df["Global_active_power"].isna().sum())
    print(f"Minute rows            : {len(df):,}")
    print(f"Missing active power   : {total_missing:,} "
          f"({total_missing / len(df):.2%})")

    power = df["Global_active_power"]

    # Daily mean power and the share of minutes that actually had a reading
    grouped = power.resample("D")
    daily_mean_kw = grouped.mean()
    coverage = grouped.count() / (24 * 60)

    daily_kwh = daily_mean_kw * 24
    # Days with too little data are unreliable -> mark as missing
    daily_kwh[coverage < MIN_DAY_COVERAGE] = np.nan
    n_bad = int(daily_kwh.isna().sum())
    print(f"Days total             : {len(daily_kwh)}")
    print(f"Days with too few data : {n_bad}")

    # Fill short gaps by time-based interpolation (only BETWEEN valid days),
    # then drop any leading/trailing days that stay empty.
    daily_kwh = daily_kwh.interpolate(method="time", limit_area="inside")
    daily_kwh = daily_kwh.dropna()
    daily_kwh.name = "consumption_kwh"
    daily_kwh.index.name = "date"
    daily_kwh = daily_kwh.asfreq("D")        # enforce a continuous daily index
    daily_kwh = daily_kwh.interpolate(method="time")

    print(f"Clean daily series     : {daily_kwh.index.min().date()} -> "
          f"{daily_kwh.index.max().date()} ({len(daily_kwh)} days)")
    return daily_kwh


def make_synthetic_daily(n_days: int = 1100) -> pd.Series:
    """Offline demo data: trend + yearly season + weekend bump + noise."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = pd.date_range("2007-01-01", periods=n_days, freq="D", name="date")
    t = np.arange(n_days)
    yearly = 6 * np.cos(2 * np.pi * (idx.dayofyear - 15) / 365.25)  # winter peak
    weekend = np.where(idx.weekday >= 5, 3.0, 0.0)
    values = 26 + yearly + weekend - 0.003 * t + rng.normal(0, 2.5, n_days)
    return pd.Series(np.clip(values, 3, None), index=idx, name="consumption_kwh")


# --------------------------------------------------------------------------- #
# 3. Exploration
# --------------------------------------------------------------------------- #
def explore(daily: pd.Series, out_dir: Path) -> None:
    """Print summary stats and save a 2x2 panel of consumption patterns."""
    print("\n=== EXPLORATION ===")
    print(daily.describe().round(2).to_string())

    df = daily.to_frame("kwh")
    df["month"] = df.index.month
    df["weekday"] = df.index.day_name()

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (a) full history + 30-day rolling mean -> long-term trend
    ax[0, 0].plot(daily.index, daily.values, lw=0.6, alpha=0.6, label="Daily")
    ax[0, 0].plot(daily.rolling(30).mean(), color="red", lw=1.5,
                  label="30-day mean")
    ax[0, 0].set(title="Daily consumption over time", ylabel="kWh")
    ax[0, 0].legend()

    # (b) average per month -> seasonality
    monthly = df.groupby("month")["kwh"].mean()
    sns.barplot(x=monthly.index, y=monthly.values, ax=ax[0, 1], color="steelblue")
    ax[0, 1].set(title="Average consumption by month (seasonality)",
                 xlabel="Month", ylabel="kWh")

    # (c) average per weekday -> weekly pattern
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
    wk = df.groupby("weekday")["kwh"].mean().reindex(order)
    sns.barplot(x=wk.index, y=wk.values, ax=ax[1, 0], color="seagreen")
    ax[1, 0].set(title="Average consumption by weekday", xlabel="", ylabel="kWh")
    ax[1, 0].tick_params(axis="x", rotation=30)

    # (d) distribution
    sns.histplot(daily, bins=40, kde=True, ax=ax[1, 1], color="purple")
    ax[1, 1].set(title="Distribution of daily consumption", xlabel="kWh")

    fig.tight_layout()
    fig.savefig(out_dir / "01_eda_patterns.png", dpi=130)
    plt.close(fig)

    print("\nAverage kWh by weekday:\n" + wk.round(2).to_string())


# --------------------------------------------------------------------------- #
# 4. Feature engineering
# --------------------------------------------------------------------------- #
def date_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features derived from the date."""
    return pd.DataFrame(
        {
            "day": index.day,
            "month": index.month,
            "weekday": index.weekday,          # Monday=0 ... Sunday=6
            "year": index.year,
            "is_weekend": (index.weekday >= 5).astype(int),
            "dayofyear": index.dayofyear,
        },
        index=index,
    )


FEATURE_COLS = (
    ["day", "month", "weekday", "year", "is_weekend", "dayofyear"]
    + [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{ROLL_WINDOW}"]
)


def build_feature_table(daily: pd.Series) -> pd.DataFrame:
    """
    Build the supervised-learning table: target + features.
    Lag features use shift() so each row only sees PAST data (no leakage).
    """
    df = date_features(daily.index)
    df["target"] = daily.values
    for k in LAGS:
        df[f"lag_{k}"] = daily.shift(k)                       # usage k days ago
    # mean of the previous 7 days (shift(1) so today is NOT included)
    df[f"roll_mean_{ROLL_WINDOW}"] = daily.shift(1).rolling(ROLL_WINDOW).mean()
    return df.dropna()                                        # drop warm-up rows


def chronological_split(df: pd.DataFrame, test_size: float):
    """First (1 - test_size) of the timeline = train, the rest = test."""
    cut = int(len(df) * (1 - test_size))
    train, test = df.iloc[:cut], df.iloc[cut:]
    print("\n=== CHRONOLOGICAL SPLIT ===")
    print(f"Train: {train.index.min().date()} -> {train.index.max().date()} "
          f"({len(train)} days)")
    print(f"Test : {test.index.min().date()} -> {test.index.max().date()} "
          f"({len(test)} days)")
    return train, test


# --------------------------------------------------------------------------- #
# 5. Models + evaluation
# --------------------------------------------------------------------------- #
def get_models() -> dict:
    return {
        # Scaling helps Linear Regression coefficients stay well-conditioned
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        # Depth/leaf limits prevent the tree from memorising the training set
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# 6. Future forecasting (recursive)
# --------------------------------------------------------------------------- #
def forecast_future(model, history: pd.Series, horizon: int) -> pd.Series:
    """
    Predict `horizon` days after the end of history, one day at a time.
    Each prediction is appended to the history so it can feed the next day's
    lag features (recursive multi-step forecasting).
    """
    values = list(history.values)
    last_date = history.index[-1]
    dates, preds = [], []

    for step in range(1, horizon + 1):
        d = last_date + pd.Timedelta(days=step)
        row = date_features(pd.DatetimeIndex([d])).iloc[0].to_dict()
        for k in LAGS:
            row[f"lag_{k}"] = values[-k]
        row[f"roll_mean_{ROLL_WINDOW}"] = float(np.mean(values[-ROLL_WINDOW:]))

        x = pd.DataFrame([row])[FEATURE_COLS]
        y_hat = max(float(model.predict(x)[0]), 0.0)   # consumption can't be < 0
        values.append(y_hat)
        dates.append(d)
        preds.append(y_hat)

    return pd.Series(preds, index=pd.DatetimeIndex(dates, name="date"),
                     name="forecast_kwh")


# --------------------------------------------------------------------------- #
# 7. Bill estimation
# --------------------------------------------------------------------------- #
def calculate_bill(kwh: float) -> float:
    """Slab tariff: each slab's units are charged at that slab's rate."""
    bill, lower = FIXED_CHARGE, 0.0
    for upper, rate in TARIFF_SLABS:
        units = max(0.0, min(kwh, upper) - lower)
        bill += units * rate
        lower = upper
    return bill


def bills_per_period(index, series_map: dict, days: int = BILLING_DAYS) -> pd.DataFrame:
    """Group a daily series into consecutive `days`-day billing periods."""
    frame = pd.DataFrame(series_map, index=index)
    block = np.arange(len(frame)) // days
    grouped = frame.groupby(block)
    full = grouped.size() == days                      # keep complete periods
    kwh = grouped.sum()[full]
    start = grouped.apply(lambda g: g.index[0].date())[full]
    bills = kwh.map(calculate_bill)
    bills.insert(0, "period_start", start)
    return bills


# --------------------------------------------------------------------------- #
# 8. Plots
# --------------------------------------------------------------------------- #
def plot_actual_vs_pred(test: pd.DataFrame, preds: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test.index, test["target"], color="black", lw=1.5, label="Actual")
    colors = {"Linear Regression": "tab:blue", "Decision Tree": "tab:orange"}
    for name, p in preds.items():
        ax.plot(test.index, p, lw=1.1, alpha=0.85, color=colors.get(name),
                label=f"{name} (MAE {mean_absolute_error(test['target'], p):.2f})")
    ax.set(title="Actual vs Predicted daily consumption (test set)",
           xlabel="Date", ylabel="kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_actual_vs_predicted.png", dpi=130)
    plt.close(fig)

    # Scatter: points on the diagonal = perfect predictions
    fig, axes = plt.subplots(1, len(preds), figsize=(11, 4.5), sharex=True,
                             sharey=True)
    axes = np.atleast_1d(axes)
    lim = [test["target"].min(), test["target"].max()]
    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(test["target"], p, s=12, alpha=0.5)
        ax.plot(lim, lim, "r--", lw=1)
        ax.set(title=name, xlabel="Actual kWh", ylabel="Predicted kWh")
    fig.tight_layout()
    fig.savefig(out_dir / "03_scatter_actual_vs_predicted.png", dpi=130)
    plt.close(fig)


def plot_forecast(history: pd.Series, forecasts: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    recent = history.iloc[-90:]
    ax.plot(recent.index, recent.values, color="black", label="History (last 90 days)")
    for name, f in forecasts.items():
        ax.plot(f.index, f.values, lw=1.5, label=f"{name} forecast")
    ax.set(title="Future consumption forecast", xlabel="Date", ylabel="kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "04_future_forecast.png", dpi=130)
    plt.close(fig)


def plot_bills(bills: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in bills.columns if c != "period_start"]
    ax = bills.set_index("period_start")[cols].plot(
        kind="bar", figsize=(12, 5), width=0.8)
    ax.set(title=f"Actual vs predicted bill per {BILLING_DAYS}-day period (test set)",
           xlabel="Period start", ylabel=f"Bill ({CURRENCY})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "05_bill_comparison.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Electricity Bill Predictor")
    ap.add_argument("--data", help="path to raw .txt or converted .csv file")
    ap.add_argument("--data-dir", default="data", help="dataset folder")
    ap.add_argument("--output-dir", default="outputs", help="results folder")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="fraction of the latest days used for testing")
    ap.add_argument("--forecast-days", type=int, default=30,
                    help="days to forecast beyond the dataset")
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated demo data (no download needed)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load + clean ----
    if args.synthetic:
        print("Using SYNTHETIC demo data (not the real UCI dataset).")
        daily = make_synthetic_daily()
    else:
        daily = clean_and_aggregate(load_raw_data(args.data, Path(args.data_dir)))
    daily.to_csv(out_dir / "daily_consumption_clean.csv")

    # ---- explore ----
    explore(daily, out_dir)

    # ---- features + split ----
    table = build_feature_table(daily)
    train, test = chronological_split(table, args.test_size)
    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

    # ---- train + evaluate ----
    print("\n=== MODEL EVALUATION (test set) ===")
    metrics, preds, fitted = {}, {}, {}

    # Naive baseline: "tomorrow = today". Any useful model should beat this.
    metrics["Naive (yesterday)"] = evaluate(y_test, X_test["lag_1"])

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
        fitted[name] = model
        metrics[name] = evaluate(y_test, preds[name])

    metrics_df = pd.DataFrame(metrics).T.round(3)
    print(metrics_df.to_string())
    metrics_df.to_csv(out_dir / "metrics.csv")
    best = min(fitted, key=lambda n: metrics[n]["MAE"])
    print(f"\nBest model by MAE: {best}")

    plot_actual_vs_pred(test, preds, out_dir)

    # ---- bills on the test period (actual vs predicted) ----
    series_map = {"Actual": y_test.values}
    series_map.update({f"{n} (pred)": p for n, p in preds.items()})
    test_bills = bills_per_period(test.index, series_map)
    print(f"\n=== BILL PER {BILLING_DAYS}-DAY PERIOD, TEST SET ({CURRENCY}) ===")
    print(test_bills.round(2).to_string(index=False))
    test_bills.to_csv(out_dir / "bills_test_period.csv", index=False)
    plot_bills(test_bills, out_dir)

    bill_mae = {c: float((test_bills[c] - test_bills["Actual"]).abs().mean())
                for c in test_bills.columns if c.endswith("(pred)")}
    print("\nMean absolute bill error per period:",
          {k: round(v, 2) for k, v in bill_mae.items()})

    # ---- future forecast: refit on ALL data, then predict ahead ----
    print(f"\n=== FORECAST: next {args.forecast_days} days ===")
    full_X, full_y = table[FEATURE_COLS], table["target"]
    forecasts, summary = {}, {}
    for name, model in get_models().items():
        model.fit(full_X, full_y)
        f = forecast_future(model, daily, args.forecast_days)
        forecasts[name] = f
        total = float(f.sum())
        summary[name] = {
            "forecast_start": str(f.index[0].date()),
            "forecast_end": str(f.index[-1].date()),
            "total_kwh": round(total, 2),
            "avg_kwh_per_day": round(total / len(f), 2),
            # slab tariff is defined per BILLING_DAYS; scale if horizon differs
            "estimated_bill": round(
                calculate_bill(total * BILLING_DAYS / len(f))
                * len(f) / BILLING_DAYS, 2),
        }
        print(f"{name:18s} {total:8.1f} kWh  ->  {CURRENCY}"
              f"{summary[name]['estimated_bill']:.2f}")

    pd.DataFrame(forecasts).to_csv(out_dir / "forecast_next_days.csv")
    plot_forecast(daily, forecasts, out_dir)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"best_model": best, "test_metrics": metrics,
             "bill_mae_per_period": bill_mae, "forecast": summary,
             "tariff": {"slabs": [[u if u != float("inf") else None, r]
                                  for u, r in TARIFF_SLABS],
                        "fixed_charge": FIXED_CHARGE,
                        "billing_days": BILLING_DAYS,
                        "currency": CURRENCY}},
            fh, indent=2)

    print(f"\nDone. All results saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
