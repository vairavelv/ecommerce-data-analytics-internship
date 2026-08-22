"""
generate_website_data.py
--------------------------
Creates a simulated, session-level website analytics dataset covering
the full conversion funnel described in Section 6 of the report:

    Session -> Landing Engagement -> Product View -> Add to Cart
    -> Checkout Started -> Payment Completed (Purchase)

Traffic sources and device types are given different drop-off
characteristics on purpose (e.g. Social has high traffic but lower
intent; Mobile converts lower than Desktop) so the traffic-source and
device analyses in the dashboard have something real to find, matching
Sections 8 and 9 of the report.

Produces:
    data/website_sessions.csv   -> one row per session

Run with:
    python data/generate_website_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(11)

N_DAYS = 90
START_DATE = datetime(2026, 5, 1)
BASE_SESSIONS_PER_DAY = 650

LANDING_PAGES = [
    "Home", "Category - Electronics", "Category - Fashion",
    "Search Results", "Campaign - Seasonal Sale", "Product Detail (Direct Link)",
]

# traffic_source -> (share of sessions, engagement quality multiplier)
TRAFFIC_SOURCES = {
    "Organic Search": dict(share=0.28, quality=1.05),
    "Direct":         dict(share=0.14, quality=1.20),
    "Paid Search":    dict(share=0.16, quality=1.10),
    "Social Media":   dict(share=0.24, quality=0.55),   # high traffic, lower intent
    "Email":          dict(share=0.10, quality=1.30),
    "Referral":       dict(share=0.08, quality=0.95),
}

# device_type -> (share of sessions, funnel quality multiplier)
DEVICES = {
    "Mobile":  dict(share=0.58, quality=0.72),   # converts lower, matches Section 9
    "Desktop": dict(share=0.34, quality=1.25),
    "Tablet":  dict(share=0.08, quality=0.95),
}

# base probability of moving from one funnel stage to the next
BASE_STAGE_RATES = dict(
    engaged=0.62,        # session -> landing engagement (i.e. 1 - bounce rate)
    product_view=0.55,   # engaged -> viewed a product
    add_to_cart=0.34,    # product view -> added to cart
    checkout_started=0.46,  # add to cart -> started checkout
    payment_completed=0.62,  # checkout started -> completed payment
)

AOV_BY_CHANNEL_NOISE = 0.25


def make_sessions():
    source_names = list(TRAFFIC_SOURCES.keys())
    source_probs = [TRAFFIC_SOURCES[s]["share"] for s in source_names]
    source_probs = np.array(source_probs) / sum(source_probs)

    device_names = list(DEVICES.keys())
    device_probs = [DEVICES[d]["share"] for d in device_names]
    device_probs = np.array(device_probs) / sum(device_probs)

    rows = []
    session_counter = 1

    for d in range(N_DAYS):
        date = START_DATE + timedelta(days=d)
        weekend_boost = 1.15 if date.weekday() >= 5 else 1.0
        n_sessions = int(RNG.poisson(BASE_SESSIONS_PER_DAY * weekend_boost))

        sources = RNG.choice(source_names, size=n_sessions, p=source_probs)
        devices = RNG.choice(device_names, size=n_sessions, p=device_probs)
        pages = RNG.choice(LANDING_PAGES, size=n_sessions)

        for i in range(n_sessions):
            source_q = TRAFFIC_SOURCES[sources[i]]["quality"]
            device_q = DEVICES[devices[i]]["quality"]
            combined_q = (source_q + device_q) / 2

            # walk the funnel stage by stage
            engaged = RNG.random() < np.clip(BASE_STAGE_RATES["engaged"] * combined_q, 0.05, 0.95)
            session_duration = max(3, RNG.normal(180 * combined_q, 60))
            pages_viewed = max(1, int(RNG.poisson(3 * combined_q))) if engaged else 1

            product_view = engaged and (RNG.random() < np.clip(BASE_STAGE_RATES["product_view"] * combined_q, 0.05, 0.95))
            add_to_cart = product_view and (RNG.random() < np.clip(BASE_STAGE_RATES["add_to_cart"] * combined_q, 0.05, 0.9))
            checkout_started = add_to_cart and (RNG.random() < np.clip(BASE_STAGE_RATES["checkout_started"] * combined_q, 0.05, 0.9))
            payment_completed = checkout_started and (RNG.random() < np.clip(BASE_STAGE_RATES["payment_completed"] * combined_q, 0.05, 0.95))

            revenue = 0.0
            if payment_completed:
                base_aov = 1350 * device_q
                revenue = round(max(200, RNG.normal(base_aov, base_aov * AOV_BY_CHANNEL_NOISE)), 2)

            rows.append({
                "session_id": f"SESS{session_counter:07d}",
                "session_date": date,
                "traffic_source": sources[i],
                "device_type": devices[i],
                "landing_page": pages[i],
                "session_duration_seconds": round(session_duration, 1),
                "pages_viewed": pages_viewed,
                "bounced": not engaged,
                "product_view": product_view,
                "added_to_cart": add_to_cart,
                "checkout_started": checkout_started,
                "payment_completed": payment_completed,
                "revenue": revenue,
            })
            session_counter += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    sessions = make_sessions()
    sessions.to_csv("data/website_sessions.csv", index=False)
    print(f"website_sessions.csv -> {len(sessions):,} rows "
          f"({sessions['session_date'].min().date()} to {sessions['session_date'].max().date()})")
