"""
generate_sales_data.py
------------------------
Creates a simulated transaction-level sales dataset with a realistic
upward trend, yearly seasonality (stronger sales near year-end), a few
promotional spikes, and some data-quality issues on purpose (duplicates,
missing/invalid revenue, cancelled orders) — so the cleaning step in
src/forecasting.py has real work to do, as described in Section 5 of
the report.

Produces:
    data/transactions.csv   -> one row per order

Run with:
    python data/generate_sales_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(21)

START_DATE = datetime(2024, 1, 1)
N_DAYS = 912          # Jan 2024 - Jun 2026 inclusive, ending on a full month boundary
BASE_ORDERS_PER_DAY = 18
BASE_AOV = 1450

PROMO_DAYS = {60, 150, 240, 330, 420, 510, 600, 690, 780, 870}  # a handful of campaign spikes


def make_transactions():
    dates = [START_DATE + timedelta(days=d) for d in range(N_DAYS)]
    rows = []
    order_counter = 1

    for d, date in enumerate(dates):
        # long-term upward trend
        trend = 1 + (d / N_DAYS) * 0.9

        # yearly seasonality: stronger sales Oct-Dec (day-of-year based sine wave, peak ~day 335)
        day_of_year = date.timetuple().tm_yday
        seasonality = 1 + 0.35 * np.sin(2 * np.pi * (day_of_year - 260) / 365) ** 3
        # gentle end-of-year lift
        if date.month in (11, 12):
            seasonality *= 1.15

        # weekday effect: weekends slightly higher
        weekday_factor = 1.15 if date.weekday() >= 5 else 1.0

        # promo spike
        promo_factor = 1.8 if d in PROMO_DAYS else 1.0

        expected_orders = BASE_ORDERS_PER_DAY * trend * seasonality * weekday_factor * promo_factor
        n_orders = max(0, int(RNG.poisson(expected_orders)))

        for _ in range(n_orders):
            aov_noise = RNG.normal(1.0, 0.25)
            revenue = max(50, BASE_AOV * trend * aov_noise * (1.3 if d in PROMO_DAYS else 1.0))
            status = "Completed"
            r = RNG.random()
            if r < 0.05:
                status = "Cancelled"
            elif r < 0.08:
                status = "Returned"

            rows.append({
                "order_id": f"ORD{order_counter:06d}",
                "order_date": date,
                "revenue": round(revenue, 2),
                "quantity": int(RNG.integers(1, 5)),
                "order_status": status,
            })
            order_counter += 1

    df = pd.DataFrame(rows)

    # inject data-quality issues on purpose
    dup_rows = df.sample(frac=0.008, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    missing_idx = df.sample(frac=0.004, random_state=2).index
    df.loc[missing_idx, "revenue"] = np.nan

    invalid_idx = df.sample(frac=0.003, random_state=3).index
    df.loc[invalid_idx, "revenue"] = -1  # invalid negative revenue

    return df.sort_values("order_date").reset_index(drop=True)


if __name__ == "__main__":
    transactions = make_transactions()
    transactions.to_csv("data/transactions.csv", index=False)
    print(f"transactions.csv -> {len(transactions):,} rows "
          f"({transactions['order_date'].min().date()} to {transactions['order_date'].max().date()})")
