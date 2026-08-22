"""
app.py
------
Pricing Strategy A/B Testing Dashboard (Streamlit)

Workflow (Section 14 of the report):
    load data -> validate experiment groups -> calculate KPIs
    -> compare control and treatment -> run statistical tests
    -> visualize results -> generate a recommendation

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import pricing_analysis as pa

st.set_page_config(page_title="Pricing A/B Testing Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return pa.load_experiment()


try:
    df = get_data()
except FileNotFoundError:
    st.error(
        "No data found. Run `python data/generate_pricing_experiment_data.py` "
        "once to create the sample CSV file, then reload this page."
    )
    st.stop()

kpis = pa.compute_group_kpis(df)
CONTROL = "Control"
GROUPS = kpis["experiment_group"].tolist()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Pricing Strategy A/B Testing")
st.caption(
    "Control vs. treatment pricing analysis on a simulated eCommerce experiment "
    "(40,000 visitors, 4 groups) — Pandas, NumPy, Matplotlib and Streamlit."
)

st.sidebar.title("Settings")
min_effect = st.sidebar.slider("Minimum practical effect (percentage points)", 0.1, 2.0, 0.5, 0.1)
st.sidebar.caption(
    "A conversion difference smaller than this is treated as not commercially "
    "meaningful, even if statistically significant (Section 9 of the report)."
)

tabs = st.tabs([
    "Overview", "Group Comparison", "Statistical Significance",
    "Revenue & Profit", "Recommendation",
])


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Experiment Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Visitors", f"{kpis['visitors'].sum():,}")
    c2.metric("Experiment Groups", f"{len(GROUPS)}")
    c3.metric("Total Revenue", f"₹{kpis['total_revenue'].sum():,.0f}")
    c4.metric("Total Estimated Profit", f"₹{kpis['estimated_profit'].sum():,.0f}")

    st.markdown("#### Group Setup")
    setup = df.groupby("experiment_group").agg(
        visitors=("visitor_id", "count"), price_shown=("price_shown", "first")
    ).reset_index().rename(columns={"price_shown": "price_shown_₹"})
    st.dataframe(setup, use_container_width=True)

    st.markdown("#### Daily Visitors by Group (randomization check)")
    daily = df.groupby([df["visit_date"].dt.date, "experiment_group"]).size().reset_index(name="visitors")
    pivot = daily.pivot(index="visit_date", columns="experiment_group", values="visitors").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    pivot.plot(ax=ax)
    ax.set_ylabel("Visitors")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    st.pyplot(fig)
    st.caption("Roughly even traffic across groups each day is a basic sanity check for random assignment.")


# ---------------------------------------------------------------------------
# Group Comparison
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Group Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Conversion Rate by Group")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["tab:orange" if g == CONTROL else "tab:blue" for g in kpis["experiment_group"]]
        ax.bar(kpis["experiment_group"], kpis["conversion_rate_%"], color=colors)
        ax.set_ylabel("Conversion Rate (%)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig)
    with col2:
        st.markdown("#### Average Order Value by Group")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(kpis["experiment_group"], kpis["avg_order_value"], color=colors)
        ax2.set_ylabel("Avg Order Value (₹)")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig2)

    st.markdown("#### Full KPI Table")
    st.dataframe(
        kpis[["experiment_group", "visitors", "purchasers", "conversion_rate_%",
              "revenue_per_visitor", "avg_order_value", "units_per_order",
              "estimated_profit", "profit_per_visitor"]],
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Statistical Significance
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Statistical Significance vs. Control")
    st.caption(
        "Two-proportion z-test for conversion rate, and Welch's t-test for order "
        "value among purchasers. Computed with NumPy — no SciPy/Statsmodels needed "
        "at these sample sizes (the normal approximation is effectively exact)."
    )

    treatment = st.selectbox("Compare Control against:", [g for g in GROUPS if g != CONTROL])

    control_df = df[df["experiment_group"] == CONTROL]
    treat_df = df[df["experiment_group"] == treatment]

    st.markdown("#### Conversion Rate — Two-Proportion Z-Test")
    z_result = pa.two_proportion_z_test(
        control_df["converted"].sum(), len(control_df),
        treat_df["converted"].sum(), len(treat_df),
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{CONTROL} rate", f"{z_result['group_a_rate_%']}%")
    c2.metric(f"{treatment} rate", f"{z_result['group_b_rate_%']}%")
    c3.metric("Difference (pp)", f"{z_result['difference_pp']:+.3f}")
    c4.metric("p-value", f"{z_result['p_value']:.5f}")

    practical = pa.practical_significance(z_result["difference_pp"], min_effect)
    stat_sig = z_result["significant_at_5%"]

    if stat_sig and practical:
        st.success(f"Statistically significant (p < 0.05) AND practically significant (≥ {min_effect} pp).")
    elif stat_sig and not practical:
        st.warning(f"Statistically significant, but the effect is smaller than the {min_effect} pp practical threshold.")
    else:
        st.error("Not statistically significant at the 5% level.")

    st.caption(f"95% confidence interval for the difference: {z_result['ci_95_%'][0]}pp to {z_result['ci_95_%'][1]}pp")

    st.markdown("#### Order Value (Among Purchasers) — Welch's T-Test")
    control_orders = control_df.loc[control_df["converted"], "revenue"]
    treat_orders = treat_df.loc[treat_df["converted"], "revenue"]
    t_result = pa.welch_t_test(control_orders, treat_orders)

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(f"{CONTROL} avg order", f"₹{t_result['group_a_mean']:.2f}")
    c6.metric(f"{treatment} avg order", f"₹{t_result['group_b_mean']:.2f}")
    c7.metric("Difference", f"₹{t_result['difference']:+.2f}")
    c8.metric("p-value", f"{t_result['p_value']:.5f}")

    if t_result["significant_at_5%"]:
        st.success("Statistically significant difference in average order value (p < 0.05).")
    else:
        st.error("No statistically significant difference in average order value.")


# ---------------------------------------------------------------------------
# Revenue & Profit
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Revenue & Profit Comparison")
    st.caption("Conversion rate alone can be misleading — revenue and estimated profit tell the fuller story.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Revenue per Visitor")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["tab:orange" if g == CONTROL else "tab:green" for g in kpis["experiment_group"]]
        ax.bar(kpis["experiment_group"], kpis["revenue_per_visitor"], color=colors)
        ax.set_ylabel("Revenue per Visitor (₹)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig)
    with col2:
        st.markdown("#### Estimated Profit per Visitor")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(kpis["experiment_group"], kpis["profit_per_visitor"], color=colors)
        ax2.set_ylabel("Profit per Visitor (₹)")
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
        st.pyplot(fig2)

    st.markdown("#### Total Revenue vs. Total Estimated Profit")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    x = range(len(kpis))
    width = 0.35
    ax3.bar([i - width / 2 for i in x], kpis["total_revenue"], width, label="Total Revenue")
    ax3.bar([i + width / 2 for i in x], kpis["estimated_profit"], width, label="Estimated Profit")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(kpis["experiment_group"], rotation=20, ha="right")
    ax3.set_ylabel("₹")
    ax3.legend()
    st.pyplot(fig3)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Recommendation")

    rec = pa.recommend_strategy(kpis, min_effect_pp=min_effect)
    st.success(f"**Recommended strategy: {rec['recommended_group']}**")
    st.write(rec["reason"])

    st.markdown("#### Candidate Comparison vs. Control")
    st.dataframe(
        rec["candidates"][["experiment_group", "conversion_lift_pp", "revenue_lift_%",
                            "profit_lift_%", "practically_significant"]],
        use_container_width=True,
    )

    st.warning(rec["caveat"])

    st.markdown("#### Risks to Confirm Before Rolling Out")
    st.markdown("""
- Was assignment actually random, and did the test run for its planned duration?
- Does the winning group hold up when checked by device type and traffic source separately?
- Are margin/cost assumptions in this simulation accurate for the real product?
- Could novelty effects fade if the price change is extended long-term?
- Are there confounding factors (promotions, shipping fee changes, competitor pricing)?
""")
