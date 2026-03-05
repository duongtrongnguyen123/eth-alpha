"""
fetch_new_data.py
Fetch ETHUSDT 30-min OHLCV from Binance from 2025-08-15 06:30 to now.
Appends to data/ETHUSDT.csv (deduplicates on timestamp).
"""
import requests
import pandas as pd
import time
from datetime import datetime, timezone

START_MS  = int(datetime(2025, 8, 15, 6, 30, tzinfo=timezone.utc).timestamp() * 1000)
END_MS    = int(datetime(2026, 3, 3, 23, 59, tzinfo=timezone.utc).timestamp() * 1000)
URL       = "https://api.binance.com/api/v3/klines"
INTERVAL  = "30m"
LIMIT     = 1000

rows = []
current = START_MS

print("Fetching ETHUSDT 30m from Binance...")
while current < END_MS:
    params = dict(symbol="ETHUSDT", interval=INTERVAL,
                  startTime=current, endTime=END_MS, limit=LIMIT)
    resp = requests.get(URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        break
    for k in data:
        rows.append({
            "timestamp": pd.to_datetime(k[0], unit="ms", utc=True).tz_localize(None),
            "open":   float(k[1]),
            "high":   float(k[2]),
            "low":    float(k[3]),
            "close":  float(k[4]),
            "volume": float(k[5]),
        })
    current = data[-1][0] + 1
    print(f"  fetched up to {rows[-1]['timestamp']}  ({len(rows)} bars total)")
    time.sleep(0.2)

new_df = pd.DataFrame(rows).set_index("timestamp")
print(f"\nFetched {len(new_df)} new bars: {new_df.index[0]} → {new_df.index[-1]}")

# load existing, append, deduplicate
existing = pd.read_csv("data/ETHUSDT.csv", parse_dates=["timestamp"], index_col="timestamp")
combined = pd.concat([existing, new_df])
combined = combined[~combined.index.duplicated(keep="last")].sort_index()

combined.to_csv("data/ETHUSDT.csv")
print(f"Saved: {len(combined)} total bars  ({existing.index[-1]} → {combined.index[-1]})")
