import os
import glob
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

csv_directory = "/private/notebooks/dataset/test"
csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))

#read, clean, and concatenate
dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df["Flow Bytes/s"] = pd.to_numeric(df["Flow Bytes/s"], errors='coerce')
    df["Flow Bytes/s"].fillna(0, inplace=True)
    dataframes.append(df)

df_all = pd.concat(dataframes, ignore_index=True)

col = "Flow Bytes/s"
series = df_all[col]

#replace infinities with NaN
series = series.replace([np.inf, -np.inf], np.nan)

#compute 1st and 99th percentiles
finite_vals = series[np.isfinite(series)]
p1 = np.percentile(finite_vals, 1)
p99 = np.percentile(finite_vals, 99)

series_clipped = series.clip(lower=p1, upper=p99)
series_clean = series_clipped.fillna(p1).where(series_clipped.notna(), p99)
df_all[col] = series_clean
start_datetime = "2000-01-01 00:00:00"
num_rows = df_all.shape[0]

#create synthetic timestamps
timestamps = pd.date_range(start=start_datetime, periods=num_rows, freq='S')
df_all = df_all.copy()
df_all["timestamp"] = timestamps
df_all.set_index("timestamp", inplace=True)

#map labels to numeric
df_all = df_all.copy()
df_all['label_numeric'] = df_all['Label'].str.strip().str.upper().apply(lambda x: 0 if x == 'BENIGN' else 1)

resampled_target = df_all['Flow Bytes/s'].resample('1T').mean()
resampled_label  = df_all['label_numeric'].resample('1T').max()

target_list = resampled_target.tolist()
label_list = resampled_label.tolist()
start_time = resampled_target.index[0]

#write .arrow file
table = pa.table({
    'start': [start_time],
    'target': [target_list],
    'labels': [label_list]
})

output_path = 'aggregated.arrow'
with pa.OSFile(output_path, 'wb') as sink:
    with ipc.new_file(sink, table.schema) as writer:
        writer.write(table)