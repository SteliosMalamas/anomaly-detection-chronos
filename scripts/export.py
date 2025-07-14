#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

# ── CONFIG ───────────────────────────────────────────────────
AGG_DATA     = "/files/private/notebooks/scripts/aggregated_data.npz"
PRED_CSV     = "/files/private/notebooks/results/predictions.csv"
OUT_CSV      = "/files/private/notebooks/results/series_flags.csv"
CONTEXT_LEN  = 50
TEST_SIZE    = 0.3
START_TIME   = "2000-01-01 00:00:00"
FREQ         = "min"

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

#load inputs
data   = np.load(AGG_DATA)
series = data["target"]
preds  = pd.read_csv(PRED_CSV)

#build timestamps
N = len(series)
ts = pd.date_range(start=START_TIME, periods=N, freq=FREQ)

#compute train & test split
trimmed = series[CONTEXT_LEN:]
total_trimmed = len(trimmed)
train_n = int((1 - TEST_SIZE) * total_trimmed)
test_n = len(preds)

if train_n + test_n != total_trimmed:
    raise ValueError(
        f"Split mismatch: {train_n} + {test_n} != {total_trimmed}"
    )

#slice the test period
start_idx = CONTEXT_LEN + train_n
end_idx   = start_idx + test_n

df_out = pd.DataFrame({
    "timestamp": ts[start_idx:end_idx],
    "value":     series[start_idx:end_idx],
    "flag":      preds["flag"].values
})

df_out.to_csv(OUT_CSV, index=False)