"""
app.py
------
Sales Trend Forecasting & Visualization Dashboard (Streamlit)

Follows the report structure:
    Overview | Exploratory Analysis | Baseline & Moving Average |
    Regression Forecast | Model Evaluation | Future Forecast | Business Insights

Run with:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import forecasting as fc

st.set_page_config(page_title="Sales Trend Forecasting Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def get_data():
    return fc.load_transactions()


try:
    raw = get_data()
except FileNotFoundError:
    st.error(
        "No data found. Run `python data/generate_sales_data.py` once to "
        "create the sample CSV file, then reload this page."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Settings")
freq_label = st.sidebar.radio("Aggregation period", ["Monthly", "Weekly", "Daily"], index=0)
freq_code = {"Monthly": "M", "Weekly": "W", "Daily": "D"}[freq_label]

ma_window = st.sidebar.slider("Moving average window (periods)", 2, 12, 3)
train_frac = st.sidebar.slider("Training split for validation", 0.5, 0.9, 0.8, 0.05)

series = fc.aggregate_sales(raw, freq=freq_code)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Sales Trend Forecasting & Visualization")
st.caption(
    "Baseline, moving-average and linear-regression trend forecasting on a "
    "simulated eCommerce sales time series — Pandas, NumPy, Matplotlib and Streamlit."
)

tabs = st.tabs([
    "Overview", "Exploratory Analysis", "Baseline & Moving Average",
    "Regression Forecast", "Model Evaluation", "Future Forecast", "Business Insights",
])


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Historical Sales Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{series['revenue'].sum():,.0f}")
    c2.metric("Total Orders", f"{series['orders'].sum():,}")
    c3.metric("Avg Order Value", f"₹{series['aov'].mean():,.0f}")
    c4.metric("Periods of History", f"{len(series):,}")

    st.markdown("#### Historical Sales — Line Chart with Trend")
    slope, intercept = fc.linear_trend_fit(series["revenue"].to_numpy())
    fitted = fc.linear_trend_fitted_values(series["revenue"].to_numpy())

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series["period"], series["revenue"], label="Actual Revenue", marker="o", markersize=3)
    ax.plot(series["period"], fitted, label="Trend Line", linestyle="--")
    ax.set_ylabel("Revenue (₹)")
    ax.set_title(f"{'Upward' if slope > 0 else 'Downward'} trend: ≈₹{slope:,.0f} per period")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Exploratory Time-Series Analysis
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Exploratory Time-Series Analysis")

    st.markdown("#### Month-over-Month Growth")
    growth = fc.mom_growth(series)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in growth["growth_%"].fillna(0)]
    ax.bar(growth["period"], growth["growth_%"].fillna(0), color=colors)
    ax.set_ylabel("% change vs. previous period")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.autofmt_xdate()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Quarterly Sales Comparison")
        q = fc.quarterly_summary(raw)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(q["quarter"], q["revenue"])
        ax2.set_ylabel("Revenue (₹)")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig2)
    with col2:
        st.markdown("#### Average Order Value Over Time")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(series["period"], series["aov"])
        ax3.set_ylabel("AOV (₹)")
        fig3.autofmt_xdate()
        st.pyplot(fig3)

    st.markdown("#### Unusually High or Low Periods")
    unusual = fc.unusual_periods(series)
    flagged = unusual[unusual["unusual"]]
    if len(flagged):
        st.dataframe(flagged[["period", "revenue", "z_score"]], use_container_width=True)
    else:
        st.caption("No periods more than 1.5 standard deviations from the mean.")


# ---------------------------------------------------------------------------
# Baseline & Moving Average
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Baseline & Moving Average")
    st.caption(
        "The baseline simply repeats the last known value. The moving average "
        f"smooths the last {ma_window} periods — a useful reference before "
        "trusting a more complex model."
    )

    values = series["revenue"].to_numpy()
    ma_line = fc.moving_average_series(values, window=ma_window)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series["period"], values, label="Actual Revenue", marker="o", markersize=3)
    ax.plot(series["period"], ma_line, label=f"{ma_window}-Period Moving Average", linestyle="--")
    ax.axhline(values[-1], color="gray", linestyle=":", label="Naive Baseline (last value)")
    ax.set_ylabel("Revenue (₹)")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Regression Forecast
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Linear Regression Trend Model")
    st.caption("Sales = intercept + slope × time. Fit on the full available history.")

    values = series["revenue"].to_numpy()
    slope, intercept = fc.linear_trend_fit(values)
    fitted = fc.linear_trend_fitted_values(values)

    c1, c2 = st.columns(2)
    c1.metric("Slope (₹ per period)", f"{slope:,.0f}")
    c2.metric("Intercept (₹)", f"{intercept:,.0f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series["period"], values, label="Actual Revenue", marker="o", markersize=3)
    ax.plot(series["period"], fitted, label="Fitted Trend Line", linestyle="--")
    ax.set_ylabel("Revenue (₹)")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

    st.info(
        f"A {'positive' if slope > 0 else 'negative'} slope indicates an "
        f"{'increasing' if slope > 0 else 'decreasing'} overall trend. "
        "This simple linear model does not capture seasonality on its own — "
        "see the Exploratory Analysis tab for seasonal patterns."
    )


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Model Evaluation — Time-Based Validation")
    st.caption(
        f"The first {train_frac:.0%} of periods are used for training; the "
        f"remaining {1 - train_frac:.0%} are held out for validation "
        "(no random shuffling, since order matters in a time series)."
    )

    bt = fc.backtest_models(series, train_frac=train_frac, ma_window=ma_window)

    if not bt:
        st.warning("Not enough data for the selected validation split. Try a smaller training fraction.")
    else:
        metrics_table = pd.DataFrame({name: v["metrics"] for name, v in bt.items()
                                       if name not in ("test_periods", "actual")}).T
        st.dataframe(metrics_table, use_container_width=True)

        st.markdown("#### Actual vs. Predicted (Validation Period)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(bt["test_periods"], bt["actual"], label="Actual", marker="o", color="black")
        for name, v in bt.items():
            if name in ("test_periods", "actual"):
                continue
            ax.plot(bt["test_periods"], v["predicted"], label=name, linestyle="--", marker="x")
        ax.set_ylabel("Revenue (₹)")
        ax.legend()
        fig.autofmt_xdate()
        st.pyplot(fig)

        st.markdown("#### Forecast Error by Period (Linear Trend Model)")
        errors = bt["actual"] - bt["Linear Trend Regression"]["predicted"]
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        colors = ["tab:red" if e < 0 else "tab:blue" for e in errors]
        ax2.bar(range(len(errors)), errors, color=colors)
        ax2.set_xticks(range(len(errors)))
        ax2.set_xticklabels([str(p.date()) for p in bt["test_periods"]], rotation=45, ha="right")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Actual − Predicted (₹)")
        st.pyplot(fig2)

        best_model = min(
            ((name, v["metrics"]["MAE"]) for name, v in bt.items() if name not in ("test_periods", "actual")),
            key=lambda x: x[1],
        )
        st.success(f"Lowest MAE on the validation period: **{best_model[0]}** (₹{best_model[1]:,.0f})")


# ---------------------------------------------------------------------------
# Future Forecast
# ---------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Future Sales Forecast")

    method_label = st.radio("Forecasting method", ["Linear Trend", "Moving Average", "Baseline"], horizontal=True)
    method_code = {"Linear Trend": "linear", "Moving Average": "moving_average", "Baseline": "baseline"}[method_label]
    periods_ahead = st.slider("Periods to forecast", 1, 12, 6)

    forecast = fc.future_forecast(series, method=method_code, periods_ahead=periods_ahead, ma_window=ma_window)
    history = series[["period", "revenue"]].assign(type="Actual")
    combined = pd.concat([history, forecast], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, group in combined.groupby("type"):
        ax.plot(group["period"], group["revenue"], label=label, marker="o", markersize=3)
    ax.axvline(series["period"].max(), color="gray", linestyle=":", label="Forecast start")
    ax.set_ylabel("Revenue (₹)")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

    st.dataframe(forecast[["period", "revenue"]], use_container_width=True)


# ---------------------------------------------------------------------------
# Business Insights
# ---------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Business Insights")

    values = series["revenue"].to_numpy()
    slope, _ = fc.linear_trend_fit(values)
    bt = fc.backtest_models(series, train_frac=train_frac, ma_window=ma_window)
    forecast = fc.future_forecast(series, method="linear", periods_ahead=3, ma_window=ma_window)
    q = fc.quarterly_summary(raw)
    best_quarter = q.loc[q["revenue"].idxmax()]

    direction = "growing" if slope > 0 else "declining"
    next_period_est = forecast["revenue"].iloc[0]

    st.markdown(f"""
- **Trend:** revenue is **{direction}** by approximately ₹{abs(slope):,.0f} per {freq_label.lower()[:-2] or 'period'},
  based on the linear regression fit over the full history.
- **Next period estimate:** the trend model projects roughly
  ₹{next_period_est:,.0f} for the next period. Treat this as one input
  alongside planned promotions and inventory — not a guarantee.
- **Strongest quarter on record:** {best_quarter['quarter']} with
  ₹{best_quarter['revenue']:,.0f} in revenue — useful context when planning
  staffing and inventory for the same period next year.
- **Model check:** on the held-out validation period, the linear trend
  model's mean absolute error was ₹{bt.get('Linear Trend Regression', {}).get('metrics', {}).get('MAE', 0):,.0f},
  compared to ₹{bt.get('Baseline (naive)', {}).get('metrics', {}).get('MAE', 0):,.0f} for the naive baseline —
  see the Model Evaluation tab for the full comparison.
- **Caveat:** this is a simple trend model on simulated data. It does not
  account for planned promotions, product launches, or external market
  conditions — combine it with business judgment, as the report recommends.
""")
