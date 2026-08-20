"""
segmentation.py
----------------
All "backend" logic for the customer segmentation dashboard.

Sections (mapped to the report):
    - load & clean                  -> Section 6: Data Preparation
    - build_customer_features        -> Section 7: Exploratory Behavior Analysis
    - rfm_scores / assign_segment    -> Section 8-9: RFM Framework & Segments
    - behavioral breakdowns          -> Section 10: Behavioral Segmentation Beyond RFM
    - hypothesis_tests                -> Section 13: Hypothesis Development
    - segment_performance             -> Section 15: Measuring Segment Performance
"""

import numpy as np
import pandas as pd

CUSTOMERS_PATH = "data/customers.csv"
TRANSACTIONS_PATH = "data/transactions.csv"


# ---------------------------------------------------------------------------
# Load & clean (Section 6: Data Preparation)
# ---------------------------------------------------------------------------

def load_customers(path: str = CUSTOMERS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signup_date"])
    df = df.drop_duplicates(subset="customer_id")
    df["preferred_device"] = df["preferred_device"].astype(str).str.strip().str.title()
    df["acquisition_channel"] = df["acquisition_channel"].astype(str).str.strip().str.title()
    return df


def load_transactions(path: str = TRANSACTIONS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    df = df.drop_duplicates(subset="transaction_id")
    df = df.dropna(subset=["customer_id"])

    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["device_type"] = df["device_type"].astype(str).str.strip().str.title()
    df["channel"] = df["channel"].astype(str).str.strip().str.title()
    df["order_status"] = df["order_status"].astype(str).str.strip().str.title()

    df["quantity"] = df["quantity"].clip(lower=1)
    df["unit_price"] = df["unit_price"].clip(lower=0)
    df["discount_pct"] = df["discount_pct"].clip(lower=0, upper=0.9)
    df["amount"] = df["amount"].clip(lower=0)

    return df


def completed(df: pd.DataFrame) -> pd.DataFrame:
    """Purchase-value calculations should exclude cancelled orders (Section 6)."""
    return df[df["order_status"] == "Completed"]


# ---------------------------------------------------------------------------
# Customer-level features (Section 7: Exploratory Behavior Analysis)
# ---------------------------------------------------------------------------

def build_customer_features(transactions: pd.DataFrame, customers: pd.DataFrame,
                             reference_date=None) -> pd.DataFrame:
    if reference_date is None:
        reference_date = transactions["transaction_date"].max() + pd.Timedelta(days=1)

    paid = completed(transactions)

    agg = paid.groupby("customer_id").agg(
        recency=("transaction_date", lambda x: (reference_date - x.max()).days),
        frequency=("transaction_id", "nunique"),
        monetary=("amount", "sum"),
    )
    agg["aov"] = agg["monetary"] / agg["frequency"]

    # discount usage rate: share of a customer's completed transactions with a discount
    discount_rate = paid.groupby("customer_id").apply(
        lambda g: (g["discount_pct"] > 0).mean(), include_groups=False
    ).rename("discount_usage_rate")

    # return rate: computed across ALL transactions (incl. returned), not just completed
    return_rate = transactions.groupby("customer_id").apply(
        lambda g: (g["order_status"] == "Returned").mean(), include_groups=False
    ).rename("return_rate")

    # favorite category = most frequent category in completed purchases
    top_category = (paid.groupby(["customer_id", "category"]).size()
                     .reset_index(name="n")
                     .sort_values("n", ascending=False)
                     .drop_duplicates("customer_id")
                     .set_index("customer_id")["category"]
                     .rename("top_category"))

    # dominant device used for purchases
    top_device = (paid.groupby(["customer_id", "device_type"]).size()
                  .reset_index(name="n")
                  .sort_values("n", ascending=False)
                  .drop_duplicates("customer_id")
                  .set_index("customer_id")["device_type"]
                  .rename("purchase_device"))

    features = (customers.set_index("customer_id")
                .join(agg, how="left")
                .join(discount_rate, how="left")
                .join(return_rate, how="left")
                .join(top_category, how="left")
                .join(top_device, how="left"))

    # customers with zero completed transactions are effectively inactive/lost
    features["frequency"] = features["frequency"].fillna(0)
    features["monetary"] = features["monetary"].fillna(0)
    features["aov"] = features["aov"].fillna(0)
    features["discount_usage_rate"] = features["discount_usage_rate"].fillna(0)
    features["return_rate"] = features["return_rate"].fillna(0)
    max_recency = int((reference_date - customers["signup_date"].min()).days)
    features["recency"] = features["recency"].fillna(max_recency)

    return features.reset_index()


# ---------------------------------------------------------------------------
# RFM scoring & segments (Section 8-9)
# ---------------------------------------------------------------------------

def rfm_scores(features: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    df = features.copy()

    # recency: lower is better -> reverse labels
    df["r_score"] = pd.qcut(df["recency"].rank(method="first"), bins,
                             labels=list(range(bins, 0, -1))).astype(int)
    df["f_score"] = pd.qcut(df["frequency"].rank(method="first"), bins,
                             labels=list(range(1, bins + 1))).astype(int)
    df["m_score"] = pd.qcut(df["monetary"].rank(method="first"), bins,
                             labels=list(range(1, bins + 1))).astype(int)
    df["rfm_score"] = df["r_score"] + df["f_score"] + df["m_score"]

    df["segment"] = df.apply(lambda row: assign_segment(row["r_score"], row["f_score"], row["m_score"]), axis=1)
    return df


def assign_segment(r: int, f: int, m: int) -> str:
    """
    Maps RFM scores (1-5 each) to the six segments named throughout the
    report: High-Value, Loyal, New, Promising, At-Risk, Inactive.
    """
    if r >= 4 and f >= 4 and m >= 4:
        return "High-Value"
    if f >= 4 and m >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New"
    if r <= 2 and f >= 3:
        return "At-Risk"
    if r <= 2 and f <= 2:
        return "Inactive"
    return "Promising"


def segment_profile(scored: pd.DataFrame) -> pd.DataFrame:
    out = (scored.groupby("segment")
           .agg(customers=("customer_id", "nunique"),
                revenue=("monetary", "sum"),
                avg_recency_days=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_order_value=("aov", "mean"),
                avg_discount_usage=("discount_usage_rate", "mean"),
                avg_return_rate=("return_rate", "mean"))
           .sort_values("revenue", ascending=False)
           .reset_index())
    out["revenue_share_%"] = (out["revenue"] / out["revenue"].sum() * 100).round(1)
    return out


# ---------------------------------------------------------------------------
# Behavioral segmentation beyond RFM (Section 10)
# ---------------------------------------------------------------------------

def category_by_segment(scored: pd.DataFrame) -> pd.DataFrame:
    out = pd.crosstab(scored["segment"], scored["top_category"], normalize="index") * 100
    return out.round(1)


def device_by_segment(scored: pd.DataFrame) -> pd.DataFrame:
    out = pd.crosstab(scored["segment"], scored["purchase_device"], normalize="index") * 100
    return out.round(1)


def channel_by_segment(scored: pd.DataFrame) -> pd.DataFrame:
    out = (scored.groupby(["segment", "acquisition_channel"])["customer_id"].nunique()
           .reset_index(name="customers"))
    return out


def new_vs_returning(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["customer_type"] = np.where(scored["frequency"] <= 1, "New (1 order)", "Returning (2+ orders)")
    out = (scored.groupby("customer_type")
           .agg(customers=("customer_id", "nunique"),
                avg_spend=("monetary", "mean"),
                avg_discount_usage=("discount_usage_rate", "mean"))
           .reset_index())
    return out


# ---------------------------------------------------------------------------
# Hypothesis testing helpers (Section 13)
# ---------------------------------------------------------------------------

def hypothesis_summary(scored: pd.DataFrame) -> list[dict]:
    """
    Returns a list of simple, testable hypotheses with the actual numbers
    computed from the simulated dataset, so each claim is backed by data
    rather than just asserted.
    """
    results = []

    # H1: At-risk / Inactive customers use more discounts on average than High-Value customers
    hv = scored.loc[scored["segment"] == "High-Value", "discount_usage_rate"].mean()
    risk = scored.loc[scored["segment"].isin(["At-Risk", "Inactive"]), "discount_usage_rate"].mean()
    results.append({
        "hypothesis": "At-risk/Inactive customers rely more on discounts than High-Value customers.",
        "evidence": f"Avg discount usage — High-Value: {hv:.0%}, At-Risk/Inactive: {risk:.0%}",
        "supported": bool(risk > hv),
    })

    # H2: Mobile purchasers have lower average order value than Desktop
    mobile_aov = scored.loc[scored["purchase_device"] == "Mobile", "aov"].mean()
    desktop_aov = scored.loc[scored["purchase_device"] == "Desktop", "aov"].mean()
    results.append({
        "hypothesis": "Mobile purchasers have a lower average order value than Desktop purchasers.",
        "evidence": f"Avg order value — Mobile: ₹{mobile_aov:,.0f}, Desktop: ₹{desktop_aov:,.0f}",
        "supported": bool(mobile_aov < desktop_aov),
    })

    # H3: High-Value customers have lower return rates than At-Risk customers
    hv_ret = scored.loc[scored["segment"] == "High-Value", "return_rate"].mean()
    risk_ret = scored.loc[scored["segment"] == "At-Risk", "return_rate"].mean()
    results.append({
        "hypothesis": "High-Value customers have a lower return rate than At-Risk customers.",
        "evidence": f"Avg return rate — High-Value: {hv_ret:.0%}, At-Risk: {risk_ret:.0%}",
        "supported": bool(hv_ret < risk_ret),
    })

    # H4: New customers are more engaged than Inactive customers
    new_eng = scored.loc[scored["segment"] == "New", "engagement_score"].mean()
    inactive_eng = scored.loc[scored["segment"] == "Inactive", "engagement_score"].mean()
    results.append({
        "hypothesis": "New customers show higher digital engagement than Inactive customers.",
        "evidence": f"Avg engagement score — New: {new_eng:.1f}, Inactive: {inactive_eng:.1f}",
        "supported": bool(new_eng > inactive_eng),
    })

    return results


# ---------------------------------------------------------------------------
# Marketing strategy lookup (Section 14)
# ---------------------------------------------------------------------------

MARKETING_STRATEGIES = {
    "High-Value": "VIP treatment: early access, loyalty rewards, minimal discounting needed to retain.",
    "Loyal": "Cross-sell and upsell campaigns; referral incentives to leverage their satisfaction.",
    "New": "Onboarding journeys, welcome offers, and education on product range to build frequency.",
    "Promising": "Targeted nudges (personalized recommendations, limited-time offers) to increase frequency.",
    "At-Risk": "Win-back campaigns with personalized discounts before they go fully inactive.",
    "Inactive": "Low-cost reactivation emails/ads; consider suppressing high-cost channels for this group.",
}


# ---------------------------------------------------------------------------
# Segment performance tracking (Section 15) - simulated month-over-month
# ---------------------------------------------------------------------------

def monthly_segment_trend(scored: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Approximates month-over-month segment revenue using each customer's
    current segment applied back across their transaction history. This
    is a simplification (a real system would recompute RFM per period) but
    is enough to demonstrate the monitoring concept from Section 15.
    """
    paid = completed(transactions).copy()
    paid["month"] = paid["transaction_date"].dt.to_period("M").astype(str)
    merged = paid.merge(scored[["customer_id", "segment"]], on="customer_id", how="left")
    out = merged.groupby(["month", "segment"])["amount"].sum().reset_index()
    return out.sort_values("month")
