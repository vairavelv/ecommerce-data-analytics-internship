"""
pricing_analysis.py
---------------------
All "backend" logic for the pricing A/B testing dashboard.

Statistical tests (two-proportion z-test, Welch's t-test) are computed
with plain NumPy + the Python standard library `math.erf` for the
normal-distribution p-value — no SciPy/Statsmodels dependency, to keep
the stack as simple as requested. With group sizes of ~10,000 (as in
this simulated experiment), the normal approximation used here is
standard practice and very close to the exact SciPy/Statsmodels result.

Sections (mapped to the report):
    - load & clean                 -> Section 11: Data Requirements
    - compute_group_kpis            -> Section 6 & 8: Metrics & Analysis
    - two_proportion_z_test          -> Section 9: Statistical Analysis Approach
    - welch_t_test                    -> Section 9: continuous metrics (order value)
    - practical_significance          -> Section 9: minimum practical effect
    - recommend_strategy              -> Section 13: Recommendations
"""

import math

import numpy as np
import pandas as pd

DATA_PATH = "data/pricing_experiment.csv"


# ---------------------------------------------------------------------------
# Load & clean (Section 11: Data Requirements)
# ---------------------------------------------------------------------------

def load_experiment(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["visit_date"])
    return clean_experiment(df)


def clean_experiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset="visitor_id")
    df["experiment_group"] = df["experiment_group"].astype(str).str.strip()
    df["converted"] = df["converted"].astype(bool)
    df["quantity"] = df["quantity"].clip(lower=0)
    df["revenue"] = df["revenue"].clip(lower=0)
    # non-converted visitors should never carry revenue/quantity
    df.loc[~df["converted"], ["revenue", "quantity"]] = 0
    return df


# ---------------------------------------------------------------------------
# Group KPIs (Section 6 & 8)
# ---------------------------------------------------------------------------

def compute_group_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.groupby("experiment_group")
           .agg(visitors=("visitor_id", "count"),
                purchasers=("converted", "sum"),
                total_revenue=("revenue", "sum"),
                total_units=("quantity", "sum"))
           .reset_index())

    out["conversion_rate_%"] = (out["purchasers"] / out["visitors"] * 100).round(3)
    out["revenue_per_visitor"] = (out["total_revenue"] / out["visitors"]).round(2)
    out["avg_order_value"] = np.where(out["purchasers"] > 0, out["total_revenue"] / out["purchasers"], 0).round(2)
    out["units_per_order"] = np.where(out["purchasers"] > 0, out["total_units"] / out["purchasers"], 0).round(2)

    # estimated gross profit using unit_cost (Section 6: profit-based metrics)
    cost_per_group = df.groupby("experiment_group").apply(
        lambda g: (g["quantity"] * g["unit_cost"]).sum(), include_groups=False
    )
    out = out.merge(cost_per_group.rename("total_cost"), on="experiment_group")
    out["estimated_profit"] = (out["total_revenue"] - out["total_cost"]).round(2)
    out["profit_per_visitor"] = (out["estimated_profit"] / out["visitors"]).round(2)

    return out.sort_values("total_revenue", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical tests (Section 9)
# ---------------------------------------------------------------------------

def _normal_p_value(z: float) -> float:
    """Two-sided p-value from a standard normal z-score using math.erf."""
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def two_proportion_z_test(conversions_a: int, n_a: int, conversions_b: int, n_b: int) -> dict:
    """
    Compares conversion rate between two groups (control = a, treatment = b).
    Returns z-statistic, p-value, and a 95% confidence interval for the
    difference in proportions (b - a).
    """
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)

    se_pooled = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled if se_pooled > 0 else 0.0
    p_value = _normal_p_value(z)

    diff = p_b - p_a
    se_diff = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "group_a_rate_%": round(p_a * 100, 3),
        "group_b_rate_%": round(p_b * 100, 3),
        "difference_pp": round(diff * 100, 3),
        "z_statistic": round(z, 3),
        "p_value": round(p_value, 5),
        "significant_at_5%": bool(p_value < 0.05),
        "ci_95_%": (round(ci_low * 100, 3), round(ci_high * 100, 3)),
    }


def welch_t_test(sample_a: np.ndarray, sample_b: np.ndarray) -> dict:
    """
    Welch's t-test for a continuous metric (e.g. order value among
    purchasers) between two groups, without assuming equal variances.
    For the large sample sizes used in this simulated experiment, the
    normal approximation for the p-value is effectively identical to
    the exact Student-t result.
    """
    a, b = np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    n_a, n_b = len(a), len(b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    t_stat = (mean_b - mean_a) / se if se > 0 else 0.0
    p_value = _normal_p_value(t_stat)

    return {
        "group_a_mean": round(mean_a, 2),
        "group_b_mean": round(mean_b, 2),
        "difference": round(mean_b - mean_a, 2),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 5),
        "significant_at_5%": bool(p_value < 0.05),
    }


def practical_significance(diff_pp: float, min_effect_pp: float = 0.5) -> bool:
    """Section 9: a result should also clear a minimum practical effect size."""
    return abs(diff_pp) >= min_effect_pp


# ---------------------------------------------------------------------------
# Recommendation (Section 13)
# ---------------------------------------------------------------------------

def recommend_strategy(kpis: pd.DataFrame, min_effect_pp: float = 0.5) -> dict:
    """
    Ranks treatment groups against Control on conversion, revenue per
    visitor and profit per visitor, and returns a simple recommendation
    that favors revenue/profit over conversion rate alone, matching
    Section 13's guidance.
    """
    control = kpis[kpis["experiment_group"] == "Control"].iloc[0]
    candidates = kpis[kpis["experiment_group"] != "Control"].copy()

    candidates["conversion_lift_pp"] = candidates["conversion_rate_%"] - control["conversion_rate_%"]
    candidates["revenue_lift_%"] = (
        (candidates["revenue_per_visitor"] - control["revenue_per_visitor"]) / control["revenue_per_visitor"] * 100
    ).round(2)
    candidates["profit_lift_%"] = (
        (candidates["profit_per_visitor"] - control["profit_per_visitor"]) / control["profit_per_visitor"] * 100
    ).round(2)
    candidates["practically_significant"] = candidates["conversion_lift_pp"].abs() >= min_effect_pp

    best = candidates.sort_values("profit_per_visitor", ascending=False).iloc[0]

    return {
        "recommended_group": best["experiment_group"],
        "reason": (
            f"Highest estimated profit per visitor (₹{best['profit_per_visitor']:.2f} vs. "
            f"₹{control['profit_per_visitor']:.2f} for Control), with a conversion "
            f"lift of {best['conversion_lift_pp']:.2f} percentage points and a revenue "
            f"per visitor of ₹{best['revenue_per_visitor']:.2f}."
        ),
        "candidates": candidates.sort_values("profit_per_visitor", ascending=False),
        "caveat": (
            "This is based on the simulated dataset only. A real rollout decision "
            "should also confirm statistical significance (see Statistical Significance "
            "tab) and account for margin, retention effects and test duration."
        ),
    }
