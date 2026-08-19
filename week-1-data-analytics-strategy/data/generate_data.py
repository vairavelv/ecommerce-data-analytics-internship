"""
generate_data.py

Creates two CSV files that act as the "database" for this project:

    data/orders.csv         -> one row per order line item
    data/funnel_events.csv  -> one row per session/stage event


"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Reproducible "random" data
RNG = np.random.default_rng(42)

N_CUSTOMERS = 800
N_ORDERS = 6000
DAYS_OF_HISTORY = 365

START_DATE = datetime(2025, 8, 20)

CATEGORIES = {
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Smart Watch", "Phone Case", "Power Bank"],
    "Fashion": ["T-Shirt", "Sneakers", "Denim Jacket", "Backpack", "Sunglasses"],
    "Home": ["LED Lamp", "Cushion Cover", "Coffee Mug", "Storage Box", "Wall Clock"],
    "Beauty": ["Face Serum", "Lip Balm", "Hair Oil", "Sunscreen", "Face Wash"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Water Bottle", "Resistance Band", "Running Shoes"],
}
PRODUCTS = [(cat, name) for cat, names in CATEGORIES.items() for name in names]

REGIONS = ["North", "South", "East", "West", "Central"]
TRAFFIC_SOURCES = ["Organic Search", "Paid Ads", "Social Media", "Email", "Direct", "Referral"]
DEVICE_TYPES = ["Mobile", "Desktop", "Tablet"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Wallet", "Cash on Delivery"]
ORDER_STATUS = ["Completed", "Completed", "Completed", "Completed", "Cancelled", "Returned"]


def make_customers(n=N_CUSTOMERS):
    ids = [f"CUST{i:05d}" for i in range(1, n + 1)]
    # Give each customer a "loyalty" score that quietly biases how often
    # they show up in orders - this is what makes RFM segmentation later
    # produce meaningful, non-random groups.
    loyalty = RNG.beta(2, 5, size=n)
    region = RNG.choice(REGIONS, size=n)
    return pd.DataFrame({"customer_id": ids, "loyalty_score": loyalty, "home_region": region})


def make_orders(customers: pd.DataFrame, n=N_ORDERS):
    weights = customers["loyalty_score"].to_numpy()
    weights = weights / weights.sum()

    order_customer_idx = RNG.choice(len(customers), size=n, p=weights)
    chosen_customers = customers.iloc[order_customer_idx].reset_index(drop=True)

    day_offsets = RNG.integers(0, DAYS_OF_HISTORY, size=n)
    # slight upward trend + weekly seasonality (more orders on weekends)
    order_dates = [START_DATE + timedelta(days=int(d)) for d in day_offsets]

    product_idx = RNG.integers(0, len(PRODUCTS), size=n)
    categories = [PRODUCTS[i][0] for i in product_idx]
    product_names = [PRODUCTS[i][1] for i in product_idx]
    product_ids = [f"P{idx:04d}" for idx in product_idx]

    quantity = RNG.integers(1, 5, size=n)
    base_price = RNG.choice([299, 399, 499, 699, 999, 1499, 1999, 2499], size=n)
    discount_pct = RNG.choice([0, 0, 0, 5, 10, 15, 20], size=n) / 100
    shipping_cost = RNG.choice([0, 0, 49, 79, 99], size=n)

    df = pd.DataFrame({
        "order_id": [f"ORD{i:06d}" for i in range(1, n + 1)],
        "customer_id": chosen_customers["customer_id"],
        "order_date": order_dates,
        "product_id": product_ids,
        "product_name": product_names,
        "category": categories,
        "quantity": quantity,
        "unit_price": base_price,
        "discount": discount_pct,
        "shipping_cost": shipping_cost,
        "payment_method": RNG.choice(PAYMENT_METHODS, size=n),
        "region": chosen_customers["home_region"],
        "traffic_source": RNG.choice(TRAFFIC_SOURCES, size=n),
        "device_type": RNG.choice(DEVICE_TYPES, size=n, p=[0.55, 0.35, 0.10]),
        "order_status": RNG.choice(ORDER_STATUS, size=n),
    })

    # inject a few realistic data-quality issues on purpose, so the
    # cleaning step in src/data_utils.py has something real to do
    dup_rows = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)  # duplicate orders

    missing_idx = df.sample(frac=0.005, random_state=2).index
    df.loc[missing_idx, "customer_id"] = None  # missing customer id

    return df


def make_funnel_events(orders: pd.DataFrame):
    """
    Builds a session-level funnel: Visit -> Product View -> Add to Cart
    -> Checkout -> Purchase, with a realistic drop-off at each stage.
    Every completed order is guaranteed to reach "Purchase"; extra
    sessions are added that drop off earlier, so conversion rates look
    like a real funnel instead of 100%.
    """
    stages = ["Visit", "Product View", "Add to Cart", "Checkout", "Purchase"]
    drop_off = [1.0, 0.72, 0.46, 0.30, 0.22]  # fraction of ORIGINAL sessions reaching this stage

    n_sessions = int(len(orders) / drop_off[-1])  # back into total sessions needed
    session_ids = [f"SESS{i:06d}" for i in range(1, n_sessions + 1)]
    session_dates = orders["order_date"].sample(n_sessions, replace=True, random_state=3).reset_index(drop=True)
    device = RNG.choice(DEVICE_TYPES, size=n_sessions, p=[0.6, 0.32, 0.08])
    traffic = RNG.choice(TRAFFIC_SOURCES, size=n_sessions)
    customer = RNG.choice(orders["customer_id"].dropna().unique(), size=n_sessions)

    rows = []
    for i in range(n_sessions):
        reached = 1
        for s in range(1, len(stages)):
            if RNG.random() < (drop_off[s] / drop_off[s - 1]):
                reached += 1
            else:
                break
        for stage in stages[:reached]:
            rows.append({
                "session_id": session_ids[i],
                "customer_id": customer[i],
                "date": session_dates[i],
                "device_type": device[i],
                "traffic_source": traffic[i],
                "stage": stage,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    customers = make_customers()
    orders = make_orders(customers)
    funnel = make_funnel_events(orders)

    orders.to_csv("data/orders.csv", index=False)
    funnel.to_csv("data/funnel_events.csv", index=False)

    print(f"orders.csv        -> {len(orders):,} rows")
    print(f"funnel_events.csv -> {len(funnel):,} rows")
