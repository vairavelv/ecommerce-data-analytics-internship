"""
website_analytics.py
----------------------
All "backend" logic for the website analytics / CRO dashboard.

Sections (mapped to the report):
    - load & clean               -> Section 19: Data Quality and Tracking
    - compute_kpis                -> Section 5: Key Performance Indicators
    - funnel_summary               -> Section 6: Conversion Funnel Framework
    - traffic_source_summary       -> Section 8: Traffic Source Analysis
    - device_summary                -> Section 9: Device Performance Analysis
    - landing_page_summary          -> Section 10: Landing Page Analysis
    - checkout_abandonment          -> Section 12: Checkout Analysis
    - hypothesis_summary             -> Sections 13 & 16: Insights & Hypotheses
"""

import numpy as np
import pandas as pd

SESSIONS_PATH = "data/website_sessions.csv"

FUNNEL_STAGES = [
    "Sessions", "Landing Engagement", "Product Views",
    "Add to Cart", "Checkout Started", "Payment Completed",
]


# ---------------------------------------------------------------------------
# Load & clean (Section 19: Data Quality and Tracking Considerations)
# ---------------------------------------------------------------------------

def load_sessions(path: str = SESSIONS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["session_date"])
    return clean_sessions(df)


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. avoid double-counting sessions
    df = df.drop_duplicates(subset="session_id")

    # 2. standardize categorical fields (consistent event/attribute naming)
    for col in ["traffic_source", "device_type", "landing_page"]:
        df[col] = df[col].astype(str).str.strip()

    # 3. boolean funnel flags should be strictly True/False
    for col in ["bounced", "product_view", "added_to_cart", "checkout_started", "payment_completed"]:
        df[col] = df[col].astype(bool)

    # 4. purchase events should only exist for completed payments (Section 19)
    df.loc[~df["payment_completed"], "revenue"] = 0.0
    df["revenue"] = df["revenue"].clip(lower=0)

    # 5. funnel consistency: a later stage implies all earlier stages happened
    df.loc[df["payment_completed"], "checkout_started"] = True
    df.loc[df["checkout_started"], "added_to_cart"] = True
    df.loc[df["added_to_cart"], "product_view"] = True
    df.loc[df["product_view"], "bounced"] = False

    return df


# ---------------------------------------------------------------------------
# KPIs (Section 5: Key Performance Indicators)
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    total_sessions = len(df)
    purchases = int(df["payment_completed"].sum())
    revenue = df["revenue"].sum()

    bounce_rate = df["bounced"].mean() * 100
    avg_session_duration = df["session_duration_seconds"].mean()
    ctr_to_product = df.loc[~df["bounced"], "product_view"].mean() * 100 if (~df["bounced"]).any() else 0
    add_to_cart_rate = df.loc[df["product_view"], "added_to_cart"].mean() * 100 if df["product_view"].any() else 0
    checkout_completion_rate = (
        df.loc[df["checkout_started"], "payment_completed"].mean() * 100
        if df["checkout_started"].any() else 0
    )
    conversion_rate = (purchases / total_sessions * 100) if total_sessions else 0
    aov = revenue / purchases if purchases else 0

    return {
        "Total Sessions": total_sessions,
        "Total Purchases": purchases,
        "Total Revenue": revenue,
        "Average Order Value": aov,
        "Conversion Rate %": conversion_rate,
        "Bounce Rate %": bounce_rate,
        "Avg Session Duration (sec)": avg_session_duration,
        "Click-Through Rate to Product %": ctr_to_product,
        "Add-to-Cart Rate %": add_to_cart_rate,
        "Checkout Completion Rate %": checkout_completion_rate,
    }


# ---------------------------------------------------------------------------
# Conversion funnel (Section 6)
# ---------------------------------------------------------------------------

def funnel_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = [
        len(df),
        int((~df["bounced"]).sum()),
        int(df["product_view"].sum()),
        int(df["added_to_cart"].sum()),
        int(df["checkout_started"].sum()),
        int(df["payment_completed"].sum()),
    ]
    out = pd.DataFrame({"stage": FUNNEL_STAGES, "sessions": counts})
    out["conversion_from_start_%"] = (out["sessions"] / out["sessions"].iloc[0] * 100).round(1)
    out["step_conversion_%"] = (out["sessions"] / out["sessions"].shift(1) * 100).round(1)
    out.loc[0, "step_conversion_%"] = 100.0
    return out


def overall_conversion_rate(df: pd.DataFrame) -> float:
    return round(df["payment_completed"].mean() * 100, 2)


# ---------------------------------------------------------------------------
# Traffic source analysis (Section 8)
# ---------------------------------------------------------------------------

def traffic_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("traffic_source")
           .agg(sessions=("session_id", "count"),
                bounce_rate=("bounced", "mean"),
                purchases=("payment_completed", "sum"),
                revenue=("revenue", "sum"))
           .reset_index())
    out["conversion_rate_%"] = (out["purchases"] / out["sessions"] * 100).round(2)
    out["bounce_rate_%"] = (out["bounce_rate"] * 100).round(1)
    out = out.drop(columns="bounce_rate").sort_values("sessions", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Device analysis (Section 9)
# ---------------------------------------------------------------------------

def device_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("device_type")
           .agg(sessions=("session_id", "count"),
                bounce_rate=("bounced", "mean"),
                purchases=("payment_completed", "sum"),
                revenue=("revenue", "sum"))
           .reset_index())
    out["conversion_rate_%"] = (out["purchases"] / out["sessions"] * 100).round(2)
    out["bounce_rate_%"] = (out["bounce_rate"] * 100).round(1)
    out = out.drop(columns="bounce_rate").sort_values("sessions", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Landing page analysis (Section 10)
# ---------------------------------------------------------------------------

def landing_page_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("landing_page")
           .agg(sessions=("session_id", "count"),
                bounce_rate=("bounced", "mean"),
                purchases=("payment_completed", "sum"),
                revenue=("revenue", "sum"))
           .reset_index())
    out["conversion_rate_%"] = (out["purchases"] / out["sessions"] * 100).round(2)
    out["bounce_rate_%"] = (out["bounce_rate"] * 100).round(1)
    out = out.drop(columns="bounce_rate").sort_values("sessions", ascending=False)
    return out


def high_traffic_low_conversion(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Flags landing pages with above-median traffic but below-median conversion (Section 10)."""
    pages = landing_page_summary(df)
    median_sessions = pages["sessions"].median()
    median_conv = pages["conversion_rate_%"].median()
    flagged = pages[(pages["sessions"] >= median_sessions) & (pages["conversion_rate_%"] < median_conv)]
    return flagged.sort_values("sessions", ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Checkout analysis (Section 12)
# ---------------------------------------------------------------------------

def checkout_abandonment(df: pd.DataFrame) -> dict:
    started = int(df["checkout_started"].sum())
    completed = int(df["payment_completed"].sum())
    abandoned = started - completed
    rate = (abandoned / started * 100) if started else 0
    return {
        "Checkouts Started": started,
        "Payments Completed": completed,
        "Checkouts Abandoned": abandoned,
        "Abandonment Rate %": round(rate, 1),
    }


def cart_to_checkout_gap(df: pd.DataFrame) -> dict:
    """Section 6: 'product engagement does not automatically translate into cart additions'."""
    carts = int(df["added_to_cart"].sum())
    checkouts = int(df["checkout_started"].sum())
    gap = carts - checkouts
    rate = (gap / carts * 100) if carts else 0
    return {
        "Added to Cart": carts,
        "Started Checkout": checkouts,
        "Carts Not Taken to Checkout": gap,
        "Cart Abandonment Rate %": round(rate, 1),
    }


# ---------------------------------------------------------------------------
# Hypotheses (Section 13 & 16) - checked against the actual simulated data
# ---------------------------------------------------------------------------

def hypothesis_summary(df: pd.DataFrame) -> list[dict]:
    results = []

    # H1: Mobile converts lower than Desktop
    mobile_cr = df.loc[df["device_type"] == "Mobile", "payment_completed"].mean() * 100
    desktop_cr = df.loc[df["device_type"] == "Desktop", "payment_completed"].mean() * 100
    results.append({
        "hypothesis": "Mobile sessions convert at a lower rate than Desktop sessions.",
        "evidence": f"Conversion rate — Mobile: {mobile_cr:.2f}%, Desktop: {desktop_cr:.2f}%",
        "supported": bool(mobile_cr < desktop_cr),
    })

    # H2: Social Media has high traffic but lower conversion than other channels
    src = traffic_source_summary(df)
    social_row = src[src["traffic_source"] == "Social Media"]
    other_avg_cr = src[src["traffic_source"] != "Social Media"]["conversion_rate_%"].mean()
    social_cr = social_row["conversion_rate_%"].iloc[0] if len(social_row) else np.nan
    results.append({
        "hypothesis": "Social Media traffic converts below the average of other channels.",
        "evidence": f"Social Media: {social_cr:.2f}%, other channels avg: {other_avg_cr:.2f}%",
        "supported": bool(social_cr < other_avg_cr) if not np.isnan(social_cr) else False,
    })

    # H3: Product engagement does not fully translate into cart additions
    gap = cart_to_checkout_gap(df)
    funnel = funnel_summary(df)
    pv_to_cart = funnel.loc[funnel["stage"] == "Add to Cart", "step_conversion_%"].iloc[0]
    results.append({
        "hypothesis": "A meaningful share of users who view products do not add anything to cart.",
        "evidence": f"Product View → Add to Cart step conversion: {pv_to_cart:.1f}%",
        "supported": bool(pv_to_cart < 60),
    })

    # H4: Checkout abandonment is a bigger loss point than landing-page bounce
    ca = checkout_abandonment(df)
    bounce_rate = df["bounced"].mean() * 100
    results.append({
        "hypothesis": "Checkout abandonment rate is high enough to be a priority alongside bounce rate.",
        "evidence": f"Checkout abandonment: {ca['Abandonment Rate %']:.1f}%, overall bounce rate: {bounce_rate:.1f}%",
        "supported": bool(ca["Abandonment Rate %"] > 20),
    })

    return results


# ---------------------------------------------------------------------------
# CRO recommendation lookup (Section 24)
# ---------------------------------------------------------------------------

CRO_RECOMMENDATIONS = {
    "High bounce rate": "Improve landing-page relevance and clarity; make sure the page matches the traffic source/campaign.",
    "Low add-to-cart rate": "Strengthen product pages — better images, clearer pricing, visible reviews and stock/delivery info.",
    "High checkout abandonment": "Simplify checkout, show delivery costs upfront, add guest checkout and more payment options.",
    "Low mobile conversion": "Audit mobile page speed, navigation, form usability and checkout flow specifically.",
    "Low channel conversion (e.g. Social)": "Revisit audience targeting and landing-page alignment for that channel before increasing spend.",
}
