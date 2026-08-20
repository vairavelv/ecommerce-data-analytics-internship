"""
generate_customer_data.py
--------------------------
Creates a simulated customer-level and transaction-level dataset, as
called for in Section 11 of the report ("Simulated Dataset and Example
Analysis"). No real customer data is used anywhere in this project.

Produces:
    data/customers.csv     -> one row per customer (signup, channel, device, engagement)
    data/transactions.csv  -> one row per transaction

Six hidden "behavior archetypes" are used only to generate realistic,
separable data (so segmentation actually finds something meaningful).
The archetype itself is NOT stored in the output files or used by the
segmentation code — segments are calculated purely from RFM + behavior
rules, the same way it would work on real data.

Run with:
    python data/generate_customer_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(7)

N_CUSTOMERS = 1000
TODAY = datetime(2026, 8, 20)

CATEGORIES = ["Electronics", "Fashion", "Home", "Beauty", "Sports"]
DEVICES = ["Mobile", "Desktop", "Tablet"]
CHANNELS = ["Organic Search", "Paid Ads", "Social Media", "Email", "Direct", "Referral"]

# archetype -> (weight, freq_range(orders/yr), recency_bias_days, monetary_mean,
#               discount_prob, return_prob, engagement_range)
ARCHETYPES = {
    "champion":   dict(weight=0.12, orders=(14, 26), recency_max=20,  spend_mean=2800, discount_p=0.25, return_p=0.03, engagement=(75, 100)),
    "loyal":      dict(weight=0.18, orders=(8, 14),  recency_max=45,  spend_mean=1600, discount_p=0.30, return_p=0.05, engagement=(55, 85)),
    "new":        dict(weight=0.15, orders=(1, 3),   recency_max=25,  spend_mean=900,  discount_p=0.40, return_p=0.05, engagement=(40, 70)),
    "promising":  dict(weight=0.15, orders=(3, 6),   recency_max=60,  spend_mean=1100, discount_p=0.35, return_p=0.06, engagement=(35, 65)),
    "at_risk":    dict(weight=0.20, orders=(5, 10),  recency_max=220, spend_mean=1300, discount_p=0.20, return_p=0.08, engagement=(15, 40)),
    "inactive":   dict(weight=0.20, orders=(1, 4),   recency_max=340, spend_mean=700,  discount_p=0.15, return_p=0.10, engagement=(5, 25)),
}


def make_customers(n=N_CUSTOMERS):
    names = list(ARCHETYPES.keys())
    probs = [ARCHETYPES[k]["weight"] for k in names]
    probs = np.array(probs) / sum(probs)
    archetype = RNG.choice(names, size=n, p=probs)

    signup_days_ago = RNG.integers(30, 730, size=n)
    signup_date = [TODAY - timedelta(days=int(d)) for d in signup_days_ago]

    engagement_score = np.array([
        RNG.uniform(*ARCHETYPES[a]["engagement"]) for a in archetype
    ]).round(1)

    preferred_device = RNG.choice(DEVICES, size=n, p=[0.55, 0.35, 0.10])
    acquisition_channel = RNG.choice(CHANNELS, size=n)

    df = pd.DataFrame({
        "customer_id": [f"CUS{i:04d}" for i in range(1, n + 1)],
        "signup_date": signup_date,
        "acquisition_channel": acquisition_channel,
        "preferred_device": preferred_device,
        "engagement_score": engagement_score,
        "_archetype": archetype,  # kept only to drive transaction generation below
    })
    return df


def make_transactions(customers: pd.DataFrame):
    rows = []
    txn_counter = 1

    for _, cust in customers.iterrows():
        a = ARCHETYPES[cust["_archetype"]]
        n_orders = int(RNG.integers(a["orders"][0], a["orders"][1] + 1))
        if n_orders == 0:
            continue

        # most recent order falls within recency_max days of "today"
        last_order_offset = int(RNG.integers(1, a["recency_max"] + 1))
        # spread earlier orders further back in time
        offsets = sorted(RNG.integers(last_order_offset, last_order_offset + 300, size=n_orders - 1)) if n_orders > 1 else []
        offsets = [last_order_offset] + offsets

        favorite_category = RNG.choice(CATEGORIES)  # category preference per customer

        for off in offsets:
            txn_date = TODAY - timedelta(days=int(off))
            category = favorite_category if RNG.random() < 0.6 else RNG.choice(CATEGORIES)
            quantity = int(RNG.integers(1, 4))
            unit_price = float(RNG.choice([299, 399, 499, 699, 999, 1499, 1999]))
            discount_pct = round(RNG.choice([0, 5, 10, 15, 20]) / 100, 2) if RNG.random() < a["discount_p"] else 0.0

            amount = quantity * unit_price * (1 - discount_pct)
            # scale roughly toward the archetype's average spend per order
            amount = amount * (a["spend_mean"] / 1200)

            status = "Completed"
            if RNG.random() < a["return_p"]:
                status = "Returned"
            elif RNG.random() < 0.04:
                status = "Cancelled"

            rows.append({
                "transaction_id": f"TXN{txn_counter:06d}",
                "customer_id": cust["customer_id"],
                "transaction_date": txn_date,
                "category": category,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "amount": round(amount, 2),
                "device_type": cust["preferred_device"] if RNG.random() < 0.8 else RNG.choice(DEVICES),
                "channel": cust["acquisition_channel"],
                "order_status": status,
            })
            txn_counter += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    customers = make_customers()
    transactions = make_transactions(customers)

    customers_out = customers.drop(columns=["_archetype"])
    customers_out.to_csv("data/customers.csv", index=False)
    transactions.to_csv("data/transactions.csv", index=False)

    print(f"customers.csv    -> {len(customers_out):,} rows")
    print(f"transactions.csv -> {len(transactions):,} rows")
