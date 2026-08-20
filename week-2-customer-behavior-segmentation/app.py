"""
app.py
------
Customer Behavior Analysis & Segmentation Dashboard (Streamlit)

Follows the report structure:
    Overview | RFM Scoring | Segments | Behavioral Analysis |
    Hypotheses | Marketing Strategies | Segment Trends

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import streamlit as st

from src import segmentation as seg

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    customers = seg.load_customers()
    transactions = seg.load_transactions()
    features = seg.build_customer_features(transactions, customers)
    scored = seg.rfm_scores(features)
    return customers, transactions, scored


try:
    customers, transactions, scored = get_data()
except FileNotFoundError:
    st.error(
        "No data found. Run `python data/generate_customer_data.py` once to "
        "create the sample CSV files, then reload this page."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")
segment_filter = st.sidebar.multiselect("Segment", sorted(scored["segment"].unique()))
channel_filter = st.sidebar.multiselect("Acquisition Channel", sorted(scored["acquisition_channel"].unique()))

view = scored.copy()
if segment_filter:
    view = view[view["segment"].isin(segment_filter)]
if channel_filter:
    view = view[view["acquisition_channel"].isin(channel_filter)]

st.sidebar.caption(f"{len(view):,} of {len(scored):,} customers match the current filters.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Customer Behavior Analysis & Segmentation")
st.caption(
    "RFM segmentation plus behavioral dimensions (category, device, discount, "
    "channel) on a simulated eCommerce customer dataset — Pandas, NumPy, "
    "Matplotlib and Streamlit."
)

tabs = st.tabs([
    "Overview", "RFM Scoring", "Segments", "Behavioral Analysis",
    "Hypotheses", "Marketing Strategies", "Segment Trends",
])


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Customer Base Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{view['customer_id'].nunique():,}")
    c2.metric("Total Revenue", f"₹{view['monetary'].sum():,.0f}")
    c3.metric("Avg Order Value", f"₹{view['aov'].mean():,.0f}")
    c4.metric("Avg Discount Usage", f"{view['discount_usage_rate'].mean():.0%}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribution of Total Spend")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(view["monetary"], bins=30)
        ax.set_xlabel("Total Spend (₹)")
        ax.set_ylabel("Customers")
        st.pyplot(fig)
    with col2:
        st.markdown("#### Distribution of Orders per Customer")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(view["frequency"], bins=20)
        ax2.set_xlabel("Number of Orders")
        ax2.set_ylabel("Customers")
        st.pyplot(fig2)


# ---------------------------------------------------------------------------
# RFM Scoring
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("RFM Scoring")
    st.caption("Each customer is scored 1 (lowest) to 5 (highest) on Recency, Frequency and Monetary value, using quintiles of the customer base.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### RFM Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(view["rfm_score"], bins=range(3, 17))
        ax.set_xlabel("Total RFM Score (3-15)")
        ax.set_ylabel("Customers")
        st.pyplot(fig)
    with col2:
        st.markdown("#### Recency vs. Monetary Value")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(view["recency"], view["monetary"], alpha=0.4, s=12)
        ax2.set_xlabel("Recency (days since last purchase)")
        ax2.set_ylabel("Monetary Value (₹)")
        st.pyplot(fig2)

    st.markdown("#### Frequency vs. Monetary Value")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.scatter(view["frequency"], view["monetary"], alpha=0.4, s=12)
    ax3.set_xlabel("Frequency (number of orders)")
    ax3.set_ylabel("Monetary Value (₹)")
    st.pyplot(fig3)

    with st.expander("View raw RFM table"):
        st.dataframe(
            view[["customer_id", "recency", "frequency", "monetary", "r_score", "f_score", "m_score", "rfm_score", "segment"]]
            .sort_values("rfm_score", ascending=False),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Proposed Customer Segments")
    st.caption("High-Value · Loyal · New · Promising · At-Risk · Inactive")

    profile = seg.segment_profile(view)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Customer Count by Segment")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(profile["segment"], profile["customers"])
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig)
    with col2:
        st.markdown("#### Revenue Contribution by Segment")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(profile["segment"], profile["revenue"])
        ax2.set_ylabel("Revenue (₹)")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig2)

    st.markdown("#### Average Order Value by Segment")
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.bar(profile["segment"], profile["avg_order_value"])
    ax3.set_ylabel("Avg Order Value (₹)")
    st.pyplot(fig3)

    st.dataframe(profile, use_container_width=True)


# ---------------------------------------------------------------------------
# Behavioral Analysis (beyond RFM)
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Behavioral Segmentation Beyond RFM")

    st.markdown("#### Category Preference by Segment (%)")
    cat = seg.category_by_segment(view)
    fig, ax = plt.subplots(figsize=(10, 4))
    cat.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("% of customers")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Device Behavior by Segment (%)")
        dev = seg.device_by_segment(view)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        dev.plot(kind="bar", stacked=True, ax=ax2)
        ax2.set_ylabel("% of customers")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig2)
    with col2:
        st.markdown("#### Discount Usage by Segment")
        profile = seg.segment_profile(view)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(profile["segment"], profile["avg_discount_usage"] * 100)
        ax3.set_ylabel("% of orders with a discount")
        plt.setp(ax3.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig3)

    st.markdown("#### New vs. Returning Customers")
    st.dataframe(seg.new_vs_returning(view), use_container_width=True)

    st.markdown("#### Acquisition Channel by Segment")
    st.dataframe(seg.channel_by_segment(view), use_container_width=True)


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Hypothesis Development")
    st.caption("Testable statements checked directly against the simulated dataset, not just asserted.")

    for h in seg.hypothesis_summary(view):
        icon = "✅" if h["supported"] else "❌"
        st.markdown(f"**{icon} {h['hypothesis']}**")
        st.caption(h["evidence"])
        st.divider()


# ---------------------------------------------------------------------------
# Marketing Strategies
# ---------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Marketing Strategies by Segment")
    for segment_name, strategy in seg.MARKETING_STRATEGIES.items():
        count = int((view["segment"] == segment_name).sum())
        st.markdown(f"**{segment_name}** ({count:,} customers) — {strategy}")


# ---------------------------------------------------------------------------
# Segment Trends
# ---------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Segment Performance Over Time")
    st.caption(
        "Monthly revenue by segment (based on each customer's current segment "
        "applied across their transaction history) — a simplified view of the "
        "monitoring process described in Section 15 of the report."
    )

    trend = seg.monthly_segment_trend(view, transactions)
    pivot = trend.pivot(index="month", columns="segment", values="amount").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    pivot.plot(ax=ax)
    ax.set_ylabel("Revenue (₹)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    st.dataframe(pivot, use_container_width=True)
