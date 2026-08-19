"""
app.py
------
eCommerce Data Analytics Dashboard (Streamlit)

"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import data_utils as du

st.set_page_config(page_title="eCommerce Analytics Dashboard", layout="wide")


# Data loading (cached so the CSVs are only read/cleaned once per session)


@st.cache_data
def get_orders():
    return du.load_orders()


@st.cache_data
def get_funnel():
    return du.load_funnel()


try:
    orders = get_orders()
    funnel = get_funnel()
except FileNotFoundError:
    st.error(
        "No data found. Run `python data/generate_data.py` once to create "
        "the sample CSV files, then reload this page."
    )
    st.stop()


# Sidebar filters


st.sidebar.title("Filters")

min_date, max_date = orders["order_date"].min(), orders["order_date"].max()
date_range = st.sidebar.date_input(
    "Order date range", value=(min_date.date(), max_date.date()),
    min_value=min_date.date(), max_value=max_date.date(),
)
regions = st.sidebar.multiselect("Region", sorted(orders["region"].unique()))
categories = st.sidebar.multiselect("Category", sorted(orders["category"].unique()))

filtered = orders.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["order_date"] >= start) & (filtered["order_date"] <= end)]
if regions:
    filtered = filtered[filtered["region"].isin(regions)]
if categories:
    filtered = filtered[filtered["category"].isin(categories)]

st.sidebar.caption(f"{len(filtered):,} order lines match the current filters.")


# Header

st.title("eCommerce Data Analytics Dashboard")
st.caption(
    "A practical, phased analytics strategy for sales, customers, products "
    "and market trends — built with Pandas, NumPy, Matplotlib and Streamlit."
)

tabs = st.tabs([
    "Overview", "Sales Trends", "Products", "Customers",
    "Funnel", "Marketing", "Regions", "Predictive", "Insights",
])


# Overview

with tabs[0]:
    st.subheader("Business Overview")
    kpis = du.compute_kpis(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{kpis['Total Revenue']:,.0f}")
    c2.metric("Completed Orders", f"{kpis['Completed Orders']:,}")
    c3.metric("Average Order Value", f"₹{kpis['Average Order Value']:,.0f}")
    c4.metric("Unique Customers", f"{kpis['Unique Customers']:,}")

    c5, c6 = st.columns(2)
    c5.metric("Cancellation Rate", f"{kpis['Cancellation Rate %']:.1f}%")
    c6.metric("Return Rate", f"{kpis['Return Rate %']:.1f}%")

    st.markdown("#### Monthly Revenue")
    trend = du.sales_trend(filtered, freq="M")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(trend["date"], trend["revenue"], marker="o")
    ax.set_ylabel("Revenue (₹)")
    ax.set_xlabel("")
    fig.autofmt_xdate()
    st.pyplot(fig)


# Sales Trends

with tabs[1]:
    st.subheader("Sales Trends")
    freq_label = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)
    freq_code = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq_label]

    trend = du.sales_trend(filtered, freq=freq_code)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trend["date"], trend["revenue"])
    ax.set_ylabel("Revenue (₹)")
    fig.autofmt_xdate()
    st.pyplot(fig)

    st.markdown("#### Revenue by Day of Week")
    dow = du.sales_by_day_of_week(filtered)
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.bar(dow["day_of_week"], dow["revenue"])
    ax2.set_ylabel("Revenue (₹)")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig2)


# Products

with tabs[2]:
    st.subheader("Product Performance")

    col1, col2 = st.columns([2, 1])
    with col1:
        n = st.slider("Top N products", 5, 20, 10)
        top = du.top_products(filtered, n=n)
        fig, ax = plt.subplots(figsize=(8, max(3, n * 0.35)))
        ax.barh(top["product_name"][::-1], top["revenue"][::-1])
        ax.set_xlabel("Revenue (₹)")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Revenue by Category")
        cat = du.category_summary(filtered)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(cat["revenue"], labels=cat["category"], autopct="%1.0f%%")
        st.pyplot(fig2)

    st.dataframe(top, use_container_width=True)


# Customers (RFM Segmentation)

with tabs[3]:
    st.subheader("Customer Segmentation (RFM)")
    st.caption(
        "Recency (days since last order), Frequency (number of orders) and "
        "Monetary value (total spend) are each scored 1-4 by quartile, "
        "then combined into a segment label."
    )

    rfm = du.rfm_segmentation(filtered)
    seg_summary = du.segment_summary(rfm)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(seg_summary["segment"], seg_summary["customers"])
        ax.set_ylabel("Customers")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(seg_summary["segment"], seg_summary["total_spend"])
        ax2.set_ylabel("Total Spend (₹)")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig2)

    st.dataframe(seg_summary, use_container_width=True)

    with st.expander("View raw RFM table"):
        st.dataframe(rfm.sort_values("monetary", ascending=False), use_container_width=True)


# Funnel

with tabs[4]:
    st.subheader("Purchase Funnel")
    st.caption("Website Visit → Product View → Add to Cart → Checkout → Purchase")

    overall = du.funnel_summary(funnel)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(overall["stage"], overall["sessions"])
    for i, row in overall.iterrows():
        ax.text(i, row["sessions"], f"{row['conversion_from_start_%']:.0f}%", ha="center", va="bottom")
    ax.set_ylabel("Sessions")
    st.pyplot(fig)

    st.markdown("#### Conversion by Device Type")
    st.dataframe(du.funnel_conversion_rate(funnel, by="device_type"), use_container_width=True)

    st.markdown("#### Conversion by Traffic Source")
    st.dataframe(du.funnel_conversion_rate(funnel, by="traffic_source"), use_container_width=True)


# Marketing

with tabs[5]:
    st.subheader("Marketing Channel Performance")
    mkt = du.marketing_summary(filtered)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(mkt["traffic_source"], mkt["revenue"])
    ax.set_ylabel("Revenue (₹)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    st.pyplot(fig)
    st.dataframe(mkt, use_container_width=True)


# Regions

with tabs[6]:
    st.subheader("Sales by Region")
    reg = du.region_summary(filtered)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(reg["region"], reg["revenue"])
    ax.set_ylabel("Revenue (₹)")
    st.pyplot(fig)
    st.dataframe(reg, use_container_width=True)


# Predictive Analytics

with tabs[7]:
    st.subheader("Revenue Forecast")
    st.caption(
        "A simple linear-trend baseline (NumPy polyfit), evaluated against "
        "a naive 'tomorrow = today' baseline, as recommended in Section 12 "
        "of the strategy report."
    )

    days_ahead = st.slider("Days to forecast", 7, 60, 14)
    forecast = du.forecast_revenue(filtered, days_ahead=days_ahead)

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, group in forecast.groupby("type"):
        ax.plot(group["date"], group["revenue"], label=label)
    ax.set_ylabel("Revenue (₹)")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

    accuracy = du.forecast_accuracy_vs_baseline(filtered)
    st.markdown("#### Backtest: last 14 days vs. naive baseline")
    st.json(accuracy)


# Insights

with tabs[8]:
    st.subheader("Insights & Recommended Actions")
    kpis = du.compute_kpis(filtered)
    top = du.top_products(filtered, n=1).iloc[0]
    seg = du.segment_summary(du.rfm_segmentation(filtered)).iloc[0]
    conv = du.funnel_conversion_rate(funnel)

    st.markdown(f"""
- **Revenue:** ₹{kpis['Total Revenue']:,.0f} from {kpis['Completed Orders']:,} completed
  orders (AOV ₹{kpis['Average Order Value']:,.0f}). Cancellation and return rates
  are {kpis['Cancellation Rate %']:.1f}% and {kpis['Return Rate %']:.1f}% — worth
  investigating if either is trending up.
- **Best performing product:** *{top['product_name']}* ({top['category']}) with
  ₹{top['revenue']:,.0f} in revenue — a candidate for featured placement or restocking priority.
- **Highest-value segment:** *{seg['segment']}* customers ({seg['customers']} people,
  ₹{seg['total_spend']:,.0f} total spend) — prioritize retention offers here.
- **Funnel:** overall Visit → Purchase conversion is {conv:.1f}%. Compare the
  device/traffic-source breakdown in the Funnel tab to find the weakest stage.
- **Next step:** review these numbers weekly, assign an owner to each KPI, and
  test one hypothesis (e.g. a mobile checkout change) before adding new metrics.
""")

    st.info(
        "This tab is intentionally short and written for a business reader — "
        "the detailed numbers live in the tabs above."
    )
