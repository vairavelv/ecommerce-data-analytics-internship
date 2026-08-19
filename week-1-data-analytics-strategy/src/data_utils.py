"""
data_utils.py
-------------
All the "backend" logic for the dashboard lives here, kept separate from
app.py so it can also be reused in a notebook or tested on its own.

Sections (these map directly to the report):
    - load & clean data                -> Section 6: Data Preparation
    - compute_kpis                     -> Section 7: KPIs
    - sales_trend                      -> Section 10: Market and Trend Analysis
    - top_products / category_summary  -> Section 7 / 10
    - rfm_segmentation                 -> Section 8: Customer Segmentation
    - funnel_summary                   -> Section 9: Purchase Funnel Analysis
    - marketing_summary / region_summary
    - forecast_revenue                 -> Section 12: Predictive Analytics
"""

import numpy as np
import pandas as pd

ORDERS_PATH = "data/orders.csv"
FUNNEL_PATH = "data/funnel_events.csv"


# ---------------------------------------------------------------------------
# Load & clean  (Section 6: Data Preparation and Quality)
# ---------------------------------------------------------------------------

def load_orders(path: str = ORDERS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    return clean_orders(df)


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. remove exact duplicate order lines
    df = df.drop_duplicates(subset=["order_id", "product_id", "customer_id"], keep="first")

    # 2. drop rows with no customer id - can't be attributed to anyone
    df = df.dropna(subset=["customer_id"])

    # 3. standardize text fields
    for col in ["category", "product_name", "region", "traffic_source", "device_type", "order_status"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # 4. guard against bad numeric values
    df["quantity"] = df["quantity"].clip(lower=1)
    df["unit_price"] = df["unit_price"].clip(lower=0)
    df["discount"] = df["discount"].clip(lower=0, upper=0.9)
    df["shipping_cost"] = df["shipping_cost"].clip(lower=0)

    # 5. derived fields
    df["gross_value"] = df["quantity"] * df["unit_price"]
    df["net_value"] = df["gross_value"] * (1 - df["discount"]) + df["shipping_cost"]
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["order_week"] = df["order_date"].dt.to_period("W").astype(str)
    df["day_of_week"] = df["order_date"].dt.day_name()

    return df


def load_funnel(path: str = FUNNEL_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["device_type"] = df["device_type"].astype(str).str.strip().str.title()
    df["traffic_source"] = df["traffic_source"].astype(str).str.strip().str.title()
    return df


def completed_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue-relevant orders only (Section 6: separate cancelled/returned)."""
    return df[df["order_status"] == "Completed"]


# ---------------------------------------------------------------------------
# KPIs (Section 7)
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    completed = completed_orders(df)
    total_orders = df["order_id"].nunique()
    completed_order_count = completed["order_id"].nunique()

    revenue = completed["net_value"].sum()
    aov = revenue / completed_order_count if completed_order_count else 0
    customers = completed["customer_id"].nunique()

    cancelled_rate = (df["order_status"] == "Cancelled").mean() * 100
    returned_rate = (df["order_status"] == "Returned").mean() * 100

    return {
        "Total Revenue": revenue,
        "Completed Orders": completed_order_count,
        "Total Orders (incl. cancelled/returned)": total_orders,
        "Average Order Value": aov,
        "Unique Customers": customers,
        "Cancellation Rate %": cancelled_rate,
        "Return Rate %": returned_rate,
    }


# ---------------------------------------------------------------------------
# Sales trend (Section 10)
# ---------------------------------------------------------------------------

_FREQ_MAP = {"D": "D", "W": "W", "M": "ME"}  # pandas 2.2+ renamed month-end to "ME"


def sales_trend(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """freq: 'D' daily, 'W' weekly, 'M' monthly"""
    completed = completed_orders(df).set_index("order_date")
    trend = completed["net_value"].resample(_FREQ_MAP.get(freq, freq)).sum().reset_index()
    trend.columns = ["date", "revenue"]
    return trend


def sales_by_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    completed = completed_orders(df)
    out = completed.groupby("day_of_week")["net_value"].sum().reindex(order).reset_index()
    out.columns = ["day_of_week", "revenue"]
    return out


# ---------------------------------------------------------------------------
# Products (Section 7 / 10)
# ---------------------------------------------------------------------------

def top_products(df: pd.DataFrame, n: int = 10, by: str = "net_value") -> pd.DataFrame:
    completed = completed_orders(df)
    out = (completed.groupby(["product_id", "product_name", "category"])
           .agg(revenue=("net_value", "sum"), units_sold=("quantity", "sum"))
           .sort_values(by="revenue" if by == "net_value" else by, ascending=False)
           .head(n)
           .reset_index())
    return out


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    completed = completed_orders(df)
    out = (completed.groupby("category")
           .agg(revenue=("net_value", "sum"), orders=("order_id", "nunique"), units=("quantity", "sum"))
           .sort_values("revenue", ascending=False)
           .reset_index())
    return out


# ---------------------------------------------------------------------------
# RFM customer segmentation (Section 8)
# ---------------------------------------------------------------------------

def rfm_segmentation(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    completed = completed_orders(df)
    if reference_date is None:
        reference_date = completed["order_date"].max() + pd.Timedelta(days=1)

    rfm = completed.groupby("customer_id").agg(
        recency=("order_date", lambda x: (reference_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("net_value", "sum"),
    ).reset_index()

    # score 1 (worst) - 4 (best) per metric using quartiles
    rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def label(row):
        if row["r_score"] >= 3 and row["f_score"] >= 3 and row["m_score"] >= 3:
            return "High-Value"
        if row["f_score"] >= 3:
            return "Loyal"
        if row["r_score"] >= 3 and row["f_score"] <= 2:
            return "New / Promising"
        if row["r_score"] <= 2 and row["f_score"] >= 3:
            return "At-Risk"
        return "Inactive"

    rfm["segment"] = rfm.apply(label, axis=1)
    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    out = (rfm.groupby("segment")
           .agg(customers=("customer_id", "nunique"), total_spend=("monetary", "sum"),
                avg_spend=("monetary", "mean"))
           .sort_values("total_spend", ascending=False)
           .reset_index())
    return out


# ---------------------------------------------------------------------------
# Funnel analysis (Section 9)
# ---------------------------------------------------------------------------

FUNNEL_STAGES = ["Visit", "Product View", "Add To Cart", "Checkout", "Purchase"]


def funnel_summary(events: pd.DataFrame, by: str = None) -> pd.DataFrame:
    """
    Overall funnel counts, or broken down by 'device_type' / 'traffic_source'
    when `by` is given.
    """
    events = events.copy()
    events["stage"] = events["stage"].str.title()

    if by is None:
        counts = events.groupby("stage")["session_id"].nunique().reindex(FUNNEL_STAGES).fillna(0)
        out = counts.reset_index()
        out.columns = ["stage", "sessions"]
        out["conversion_from_start_%"] = (out["sessions"] / out["sessions"].iloc[0] * 100).round(1)
        return out

    counts = events.groupby([by, "stage"])["session_id"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=FUNNEL_STAGES, fill_value=0)
    return counts.reset_index()


def funnel_conversion_rate(events: pd.DataFrame, by: str = None) -> pd.DataFrame:
    """Visit -> Purchase conversion rate overall, or split by device/traffic source."""
    summary = funnel_summary(events, by=by)
    if by is None:
        rate = summary.loc[summary["stage"] == "Purchase", "sessions"].values[0] / \
               summary.loc[summary["stage"] == "Visit", "sessions"].values[0] * 100
        return round(rate, 1)
    summary["conversion_rate_%"] = (summary["Purchase"] / summary["Visit"].replace(0, np.nan) * 100).round(1)
    return summary[[by, "Visit", "Purchase", "conversion_rate_%"]]


# ---------------------------------------------------------------------------
# Marketing & regions
# ---------------------------------------------------------------------------

def marketing_summary(df: pd.DataFrame) -> pd.DataFrame:
    completed = completed_orders(df)
    out = (completed.groupby("traffic_source")
           .agg(revenue=("net_value", "sum"), orders=("order_id", "nunique"),
                customers=("customer_id", "nunique"))
           .sort_values("revenue", ascending=False)
           .reset_index())
    return out


def region_summary(df: pd.DataFrame) -> pd.DataFrame:
    completed = completed_orders(df)
    out = (completed.groupby("region")
           .agg(revenue=("net_value", "sum"), orders=("order_id", "nunique"),
                customers=("customer_id", "nunique"))
           .sort_values("revenue", ascending=False)
           .reset_index())
    return out


# ---------------------------------------------------------------------------
# Predictive analytics (Section 12) - simple, no ML libraries required
# ---------------------------------------------------------------------------

def forecast_revenue(df: pd.DataFrame, days_ahead: int = 14) -> pd.DataFrame:
    """
    A simple linear-trend forecast using numpy.polyfit as the baseline
    described in the report (student-level regression, evaluated against
    a naive baseline). Good enough to demonstrate the workflow; a real
    implementation would swap this for scikit-learn.
    """
    trend = sales_trend(df, freq="D")
    trend = trend.sort_values("date").reset_index(drop=True)
    x = np.arange(len(trend))
    y = trend["revenue"].to_numpy()

    # linear fit: revenue = m*x + c
    m, c = np.polyfit(x, y, 1)
    future_x = np.arange(len(trend), len(trend) + days_ahead)
    future_dates = pd.date_range(trend["date"].max() + pd.Timedelta(days=1), periods=days_ahead)
    predicted = m * future_x + c
    predicted = np.clip(predicted, a_min=0, a_max=None)

    forecast_df = pd.DataFrame({"date": future_dates, "revenue": predicted, "type": "Forecast"})
    history_df = trend.rename(columns={"revenue": "revenue"}).assign(type="Actual")
    return pd.concat([history_df, forecast_df], ignore_index=True)


def forecast_accuracy_vs_baseline(df: pd.DataFrame) -> dict:
    """
    Compares the linear-trend model against a naive baseline (predict
    tomorrow = same as today) on the last 14 days of known history, as
    the report recommends evaluating against a simple baseline.
    """
    trend = sales_trend(df, freq="D").sort_values("date").reset_index(drop=True)
    if len(trend) < 30:
        return {"note": "Not enough history to backtest."}

    train = trend.iloc[:-14]
    test = trend.iloc[-14:]

    x = np.arange(len(train))
    m, c = np.polyfit(x, train["revenue"].to_numpy(), 1)
    future_x = np.arange(len(train), len(train) + len(test))
    model_pred = np.clip(m * future_x + c, 0, None)

    naive_pred = np.full(len(test), train["revenue"].iloc[-1])

    actual = test["revenue"].to_numpy()
    model_mae = np.mean(np.abs(actual - model_pred))
    naive_mae = np.mean(np.abs(actual - naive_pred))

    return {
        "Linear Trend MAE": round(float(model_mae), 2),
        "Naive Baseline MAE": round(float(naive_mae), 2),
        "Model Beats Baseline": bool(model_mae < naive_mae),
    }
