"""
app.py
------
Website Analytics & Conversion Rate Optimization Dashboard (Streamlit)

Implements Appendix A's suggested dashboard components:
    KPI cards (sessions, conversion rate, revenue, AOV) | conversion
    funnel chart | traffic-source conversion chart | device conversion
    comparison | landing-page performance table | checkout abandonment
    indicator | filters for date, device and traffic source.

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import website_analytics as wa

st.set_page_config(page_title="Website Analytics & CRO Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return wa.load_sessions()


try:
    sessions = get_data()
except FileNotFoundError:
    st.error(
        "No data found. Run `python data/generate_website_data.py` once to "
        "create the sample CSV file, then reload this page."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar filters (Appendix A: filters for date, device, traffic source)
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")

min_date, max_date = sessions["session_date"].min(), sessions["session_date"].max()
date_range = st.sidebar.date_input(
    "Session date range", value=(min_date.date(), max_date.date()),
    min_value=min_date.date(), max_value=max_date.date(),
)
device_filter = st.sidebar.multiselect("Device", sorted(sessions["device_type"].unique()))
source_filter = st.sidebar.multiselect("Traffic source", sorted(sessions["traffic_source"].unique()))

df = sessions.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df[(df["session_date"] >= start) & (df["session_date"] <= end)]
if device_filter:
    df = df[df["device_type"].isin(device_filter)]
if source_filter:
    df = df[df["traffic_source"].isin(source_filter)]

st.sidebar.caption(f"{len(df):,} of {len(sessions):,} sessions match the current filters.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Website Analytics & Conversion Rate Optimization")
st.caption(
    "KPIs, conversion funnel, traffic-source and device analysis on a "
    "simulated eCommerce website dataset — Pandas, NumPy, Matplotlib and Streamlit."
)

tabs = st.tabs([
    "Overview", "Conversion Funnel", "Traffic Sources", "Devices",
    "Landing Pages", "Checkout", "Hypotheses & Recommendations",
])


# ---------------------------------------------------------------------------
# Overview (Appendix A: KPI cards)
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Key Performance Indicators")
    kpis = wa.compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions", f"{kpis['Total Sessions']:,}")
    c2.metric("Conversion Rate", f"{kpis['Conversion Rate %']:.2f}%")
    c3.metric("Total Revenue", f"₹{kpis['Total Revenue']:,.0f}")
    c4.metric("Average Order Value", f"₹{kpis['Average Order Value']:,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Bounce Rate", f"{kpis['Bounce Rate %']:.1f}%")
    c6.metric("Avg Session Duration", f"{kpis['Avg Session Duration (sec)']:.0f} sec")
    c7.metric("Add-to-Cart Rate", f"{kpis['Add-to-Cart Rate %']:.1f}%")
    c8.metric("Checkout Completion Rate", f"{kpis['Checkout Completion Rate %']:.1f}%")

    st.markdown("#### Daily Sessions vs. Purchases")
    daily = df.groupby(df["session_date"].dt.date).agg(
        sessions=("session_id", "count"), purchases=("payment_completed", "sum")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(daily["session_date"], daily["sessions"], label="Sessions")
    ax2 = ax.twinx()
    ax2.plot(daily["session_date"], daily["purchases"], label="Purchases", color="tab:orange")
    ax.set_ylabel("Sessions")
    ax2.set_ylabel("Purchases")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.autofmt_xdate()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Conversion Funnel (Section 6, Appendix A)
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Conversion Funnel")
    st.caption("Sessions → Landing Engagement → Product Views → Add to Cart → Checkout Started → Payment Completed")

    funnel = wa.funnel_summary(df)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(funnel["stage"][::-1], funnel["sessions"][::-1])
    for i, row in funnel[::-1].reset_index(drop=True).iterrows():
        ax.text(row["sessions"], i, f"  {row['sessions']:,} ({row['conversion_from_start_%']:.1f}%)", va="center")
    ax.set_xlabel("Sessions")
    st.pyplot(fig)

    st.markdown("#### Step-by-Step Conversion")
    st.dataframe(funnel, use_container_width=True)

    st.info(
        f"Overall Sessions → Purchase conversion rate: **{wa.overall_conversion_rate(df):.2f}%**. "
        "The largest single drop identifies where to focus first."
    )


# ---------------------------------------------------------------------------
# Traffic Sources (Section 8, Appendix A)
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Traffic Source Analysis")
    src = wa.traffic_source_summary(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Sessions by Traffic Source")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(src["traffic_source"], src["sessions"])
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)
    with col2:
        st.markdown("#### Conversion Rate by Traffic Source")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(src["traffic_source"], src["conversion_rate_%"])
        ax2.set_ylabel("Conversion Rate (%)")
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig2)

    st.dataframe(src, use_container_width=True)


# ---------------------------------------------------------------------------
# Devices (Section 9, Appendix A)
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Device Performance Analysis")
    dev = wa.device_summary(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Conversion Rate by Device")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(dev["device_type"], dev["conversion_rate_%"])
        ax.set_ylabel("Conversion Rate (%)")
        st.pyplot(fig)
    with col2:
        st.markdown("#### Bounce Rate by Device")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(dev["device_type"], dev["bounce_rate_%"])
        ax2.set_ylabel("Bounce Rate (%)")
        st.pyplot(fig2)

    st.dataframe(dev, use_container_width=True)


# ---------------------------------------------------------------------------
# Landing Pages (Section 10, Appendix A)
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Landing Page Performance")
    pages = wa.landing_page_summary(df)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(pages["landing_page"], pages["conversion_rate_%"])
    ax.set_ylabel("Conversion Rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    st.pyplot(fig)

    st.markdown("#### Full Landing Page Table")
    st.dataframe(pages, use_container_width=True)

    flagged = wa.high_traffic_low_conversion(df)
    if len(flagged):
        st.warning(
            "High traffic but below-median conversion — worth investigating first: "
            + ", ".join(flagged["landing_page"].tolist())
        )


# ---------------------------------------------------------------------------
# Checkout (Section 12, Appendix A: checkout abandonment indicator)
# ---------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Checkout Analysis")

    ca = wa.checkout_abandonment(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Checkouts Started", f"{ca['Checkouts Started']:,}")
    c2.metric("Payments Completed", f"{ca['Payments Completed']:,}")
    c3.metric("Checkouts Abandoned", f"{ca['Checkouts Abandoned']:,}")
    c4.metric("Abandonment Rate", f"{ca['Abandonment Rate %']:.1f}%")

    gap = wa.cart_to_checkout_gap(df)
    st.markdown("#### Cart → Checkout Gap")
    st.caption("Product engagement does not automatically translate into cart additions, and not every cart reaches checkout.")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Added to Cart", f"{gap['Added to Cart']:,}")
    c6.metric("Started Checkout", f"{gap['Started Checkout']:,}")
    c7.metric("Carts Not Taken to Checkout", f"{gap['Carts Not Taken to Checkout']:,}")
    c8.metric("Cart Abandonment Rate", f"{gap['Cart Abandonment Rate %']:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Added to Cart", "Checkout Started", "Payment Completed"],
           [gap["Added to Cart"], gap["Started Checkout"], ca["Payments Completed"]])
    ax.set_ylabel("Sessions")
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Hypotheses & Recommendations (Sections 13, 16, 24)
# ---------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Hypothesis Development")
    st.caption("Testable statements checked directly against the simulated dataset, not just asserted.")

    for h in wa.hypothesis_summary(df):
        icon = "✅" if h["supported"] else "❌"
        st.markdown(f"**{icon} {h['hypothesis']}**")
        st.caption(h["evidence"])
        st.divider()

    st.subheader("CRO Recommendations by Issue Type")
    for issue, rec in wa.CRO_RECOMMENDATIONS.items():
        st.markdown(f"**{issue}** — {rec}")

    st.info(
        "These are analytical signals rather than final conclusions. "
        "The report recommends validating any change with a controlled A/B test "
        "before rolling it out (Section 17)."
    )
