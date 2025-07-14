import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ARROW_FILE = 'aggregated.arrow'
NPZ_FILE = 'aggregated_data.npz'

with pa.memory_map(ARROW_FILE, 'r') as source:
    reader = ipc.RecordBatchFileReader(source)
    table = reader.read_all()

df = table.to_pandas()

#extract labels
target = np.array(df['target'].iloc[0])
labels = np.array(df['labels'].iloc[0])

np.savez(NPZ_FILE,
         target=target,
         labels=labels)
