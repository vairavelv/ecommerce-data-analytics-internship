"""
generate_pricing_experiment_data.py
--------------------------------------
Creates a simulated visitor-level A/B(/n) pricing experiment dataset,
matching the scenario described in Section 7 of the report: 40,000
eligible visitors randomly split into four groups of 10,000, testing a
control price against a moderate discount, a premium price, and a
value offer.

Produces:
    data/pricing_experiment.csv   -> one row per visitor

Run with:
    python data/generate_pricing_experiment_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(5)

VISITORS_PER_GROUP = 10_000
TEST_START = datetime(2026, 6, 1)
TEST_DAYS = 14

BASE_COST_PER_UNIT = 600  # what the product actually costs the business to fulfil

# group -> (price shown, true conversion probability, avg quantity per order)
GROUPS = {
    "Control":            dict(price=1000, conversion_p=0.0420, avg_qty=1.05),
    "Moderate Discount":  dict(price=850,  conversion_p=0.0530, avg_qty=1.25),
    "Premium":            dict(price=1300, conversion_p=0.0290, avg_qty=1.02),
    "Value Offer":        dict(price=950,  conversion_p=0.0530, avg_qty=1.15),
}

DEVICES = ["Mobile", "Desktop", "Tablet"]
TRAFFIC_SOURCES = ["Organic Search", "Paid Ads", "Social Media", "Email", "Direct"]


def make_experiment_data():
    rows = []
    visitor_counter = 1

    for group_name, cfg in GROUPS.items():
        for _ in range(VISITORS_PER_GROUP):
            visit_day = int(RNG.integers(0, TEST_DAYS))
            visit_date = TEST_START + timedelta(days=visit_day)
            device = RNG.choice(DEVICES, p=[0.55, 0.35, 0.10])
            source = RNG.choice(TRAFFIC_SOURCES)

            converted = RNG.random() < cfg["conversion_p"]
            quantity = 0
            revenue = 0.0
            discount_amount = 0.0

            if converted:
                quantity = max(1, int(round(RNG.normal(cfg["avg_qty"], 0.3))))
                unit_price_noise = RNG.normal(1.0, 0.04)
                revenue = round(cfg["price"] * quantity * unit_price_noise, 2)
                discount_amount = round(max(0, (GROUPS["Control"]["price"] - cfg["price"]) * quantity), 2)

            rows.append({
                "visitor_id": f"VIS{visitor_counter:06d}",
                "experiment_group": group_name,
                "price_shown": cfg["price"],
                "visit_date": visit_date,
                "device_type": device,
                "traffic_source": source,
                "converted": converted,
                "quantity": quantity,
                "revenue": revenue,
                "discount_amount": discount_amount,
                "unit_cost": BASE_COST_PER_UNIT,
            })
            visitor_counter += 1

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)  # shuffle assignment order


if __name__ == "__main__":
    df = make_experiment_data()
    df.to_csv("data/pricing_experiment.csv", index=False)
    print(f"pricing_experiment.csv -> {len(df):,} rows across {df['experiment_group'].nunique()} groups")
    print(df.groupby("experiment_group")["converted"].mean().mul(100).round(2))
