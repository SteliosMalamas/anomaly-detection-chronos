import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import sys

BENIGN_CSV = "benign.csv"
OUTPUT_ARROW = "benign.arrow"
TARGET_COL = "Flow Bytes/s"
PCT_LOW, PCT_HIGH = 1, 99
TIMESTAMP_START = "2000-01-01 00:00:00"
TIMESTAMP_FREQ = "S" #per-second synthetic timestamp
RESAMPLE_FREQ = "1min"

#load .csv
try:
    df = pd.read_csv(BENIGN_CSV)
except Exception as e:
    sys.exit(1)
df.columns = df.columns.str.strip()

#filter only benign rows (just in case)
if "Label" in df.columns:
    df = df[df["Label"].str.strip().str.upper() == "BENIGN"].copy()

#convert to numeric & replace 'inf' with 'Nan'
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df[TARGET_COL].replace([np.inf, -np.inf], np.nan, inplace=True)

#compute low & high percentile on finite values
finite_vals = df[TARGET_COL].dropna().values
if finite_vals.size == 0:
    sys.exit(1)

low_cap  = np.percentile(finite_vals, PCT_LOW)
high_cap = np.percentile(finite_vals, PCT_HIGH)

#clip and fill 'NaN's
df[TARGET_COL] = df[TARGET_COL].clip(lower=low_cap, upper=high_cap)
df[TARGET_COL].fillna(low_cap, inplace=True)

#generate synthetic timestamps
num_rows = len(df)
timestamps = pd.date_range(start=TIMESTAMP_START, periods=num_rows, freq=TIMESTAMP_FREQ)
df = df.copy()
df["timestamp"] = timestamps
df.set_index("timestamp", inplace=True)

#resample & write IPC .arrow file
resampled = df[TARGET_COL].resample(RESAMPLE_FREQ).mean()

start_time = resampled.index[0]
target_list = resampled.tolist()
n_intervals = len(target_list)

table = pa.table({
    "start": [start_time],
    "target": [target_list]
})

with pa.OSFile(OUTPUT_ARROW, "wb") as sink:
    with ipc.new_file(sink, table.schema) as writer:
        writer.write(table)