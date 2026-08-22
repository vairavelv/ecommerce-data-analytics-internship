"""
forecasting.py
--------------
All "backend" logic for the sales forecasting dashboard.

Sections (mapped to the report):
    - load & clean                    -> Section 5: Data Familiarization and Preparation
    - aggregate_sales                 -> Section 5: aggregate to daily/weekly/monthly
    - mom_growth / quarterly_summary   -> Section 6: Exploratory Time-Series Analysis
    - baseline_forecast                -> Section 8.1: Historical Baseline
    - moving_average_forecast          -> Section 8.2: Moving Average
    - linear_trend_forecast            -> Section 8.3: Linear Regression Trend Model
    - time_based_split / evaluate      -> Section 9: Model Evaluation
    - backtest_models                  -> compares all three methods on a held-out period
"""

import numpy as np
import pandas as pd

TRANSACTIONS_PATH = "data/transactions.csv"

_FREQ_MAP = {"D": "D", "W": "W", "M": "ME"}  # pandas 2.2+ renamed month-end to "ME"


# ---------------------------------------------------------------------------
# Load & clean (Section 5: Data Familiarization and Preparation)
# ---------------------------------------------------------------------------

def load_transactions(path: str = TRANSACTIONS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    return clean_transactions(df)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. remove exact duplicate order records
    df = df.drop_duplicates(subset=["order_id"])

    # 2. drop rows with missing or invalid (negative) revenue
    df = df.dropna(subset=["revenue"])
    df = df[df["revenue"] >= 0]

    # 3. standardize status text
    df["order_status"] = df["order_status"].astype(str).str.strip().str.title()

    # 4. sort chronologically
    df = df.sort_values("order_date").reset_index(drop=True)

    return df


def completed(df: pd.DataFrame) -> pd.DataFrame:
    """Separate cancelled/returned orders from completed sales (Section 5)."""
    return df[df["order_status"] == "Completed"]


# ---------------------------------------------------------------------------
# Aggregation (Section 5: aggregate transaction-level data into a time series)
# ---------------------------------------------------------------------------

def aggregate_sales(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """
    Aggregates completed orders into a regular time series.
    freq: 'D' daily, 'W' weekly, 'M' monthly.
    Missing periods are filled with 0 revenue / 0 orders rather than
    silently dropped, since the report notes missing periods should be
    investigated rather than ignored.
    """
    paid = completed(df).set_index("order_date")
    resampled = paid.resample(_FREQ_MAP.get(freq, freq)).agg(
        revenue=("revenue", "sum"),
        orders=("order_id", "count"),
    ).reset_index()
    resampled = resampled.rename(columns={"order_date": "period"})
    resampled["aov"] = np.where(resampled["orders"] > 0, resampled["revenue"] / resampled["orders"], 0)
    return resampled


# ---------------------------------------------------------------------------
# Exploratory time-series analysis (Section 6)
# ---------------------------------------------------------------------------

def mom_growth(series_df: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month (or period-over-period) percentage change in revenue."""
    out = series_df.copy()
    out["growth_%"] = out["revenue"].pct_change() * 100
    return out


def quarterly_summary(df: pd.DataFrame) -> pd.DataFrame:
    paid = completed(df).copy()
    paid["quarter"] = paid["order_date"].dt.to_period("Q").astype(str)
    out = paid.groupby("quarter").agg(revenue=("revenue", "sum"), orders=("order_id", "count")).reset_index()
    return out


def unusual_periods(series_df: pd.DataFrame, n_std: float = 1.5) -> pd.DataFrame:
    """Flags periods where revenue is more than n_std standard deviations from the mean."""
    out = series_df.copy()
    mean, std = out["revenue"].mean(), out["revenue"].std()
    out["z_score"] = (out["revenue"] - mean) / std if std > 0 else 0
    out["unusual"] = out["z_score"].abs() > n_std
    return out


# ---------------------------------------------------------------------------
# Forecasting methods (Section 8)
# ---------------------------------------------------------------------------

def baseline_forecast(series: np.ndarray, periods_ahead: int) -> np.ndarray:
    """Section 8.1: naive forecast = repeat the last known value."""
    return np.full(periods_ahead, series[-1])


def moving_average_forecast(series: np.ndarray, window: int = 3, periods_ahead: int = 1) -> np.ndarray:
    """
    Section 8.2: forecast each future period as the average of the
    trailing `window` periods, rolling the forecast forward.
    """
    history = list(series)
    forecasts = []
    for _ in range(periods_ahead):
        avg = np.mean(history[-window:])
        forecasts.append(avg)
        history.append(avg)
    return np.array(forecasts)


def moving_average_series(series: np.ndarray, window: int = 3) -> np.ndarray:
    """The moving-average line overlaid on historical sales (visualization)."""
    return pd.Series(series).rolling(window=window, min_periods=1).mean().to_numpy()


def linear_trend_fit(series: np.ndarray):
    """Section 8.3: fit Sales = intercept + slope * time. Returns (slope, intercept)."""
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    return slope, intercept


def linear_trend_forecast(series: np.ndarray, periods_ahead: int) -> np.ndarray:
    slope, intercept = linear_trend_fit(series)
    future_x = np.arange(len(series), len(series) + periods_ahead)
    forecast = slope * future_x + intercept
    return np.clip(forecast, a_min=0, a_max=None)


def linear_trend_fitted_values(series: np.ndarray) -> np.ndarray:
    """In-sample fitted trend line, for plotting against actual history."""
    slope, intercept = linear_trend_fit(series)
    x = np.arange(len(series))
    return slope * x + intercept


# ---------------------------------------------------------------------------
# Model evaluation (Section 9)
# ---------------------------------------------------------------------------

def time_based_split(series_df: pd.DataFrame, train_frac: float = 0.8):
    """Earlier periods for training, later periods held out for validation."""
    split_idx = max(1, int(len(series_df) * train_frac))
    return series_df.iloc[:split_idx].reset_index(drop=True), series_df.iloc[split_idx:].reset_index(drop=True)


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    nonzero = actual != 0
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else np.nan

    return {"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2), "MAPE_%": round(float(mape), 2)}


def backtest_models(series_df: pd.DataFrame, train_frac: float = 0.8, ma_window: int = 3) -> dict:
    """
    Runs all three methods (baseline, moving average, linear trend) on a
    time-based validation split and returns their error metrics side by
    side, exactly as Section 9 recommends.
    """
    train, test = time_based_split(series_df, train_frac)
    if len(test) == 0:
        return {}

    train_values = train["revenue"].to_numpy()
    actual = test["revenue"].to_numpy()
    n = len(test)

    baseline_pred = baseline_forecast(train_values, n)
    ma_pred = moving_average_forecast(train_values, window=min(ma_window, len(train_values)), periods_ahead=n)
    trend_pred = linear_trend_forecast(train_values, n)

    return {
        "Baseline (naive)": {"predicted": baseline_pred, "metrics": evaluate_forecast(actual, baseline_pred)},
        f"Moving Average ({ma_window}-period)": {"predicted": ma_pred, "metrics": evaluate_forecast(actual, ma_pred)},
        "Linear Trend Regression": {"predicted": trend_pred, "metrics": evaluate_forecast(actual, trend_pred)},
        "test_periods": test["period"].tolist(),
        "actual": actual,
    }


# ---------------------------------------------------------------------------
# Future forecast (beyond the historical data)
# ---------------------------------------------------------------------------

def future_forecast(series_df: pd.DataFrame, method: str = "linear", periods_ahead: int = 6,
                     ma_window: int = 3) -> pd.DataFrame:
    values = series_df["revenue"].to_numpy()

    if method == "baseline":
        preds = baseline_forecast(values, periods_ahead)
    elif method == "moving_average":
        preds = moving_average_forecast(values, window=ma_window, periods_ahead=periods_ahead)
    else:
        preds = linear_trend_forecast(values, periods_ahead)

    last_period = series_df["period"].max()
    # infer the step size from the existing series
    if len(series_df) >= 2:
        step = series_df["period"].iloc[-1] - series_df["period"].iloc[-2]
    else:
        step = pd.Timedelta(days=30)
    future_periods = [last_period + step * (i + 1) for i in range(periods_ahead)]

    return pd.DataFrame({"period": future_periods, "revenue": preds, "type": "Forecast"})
