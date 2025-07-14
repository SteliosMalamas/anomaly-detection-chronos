#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    confusion_matrix,
)
from scipy.ndimage import binary_dilation, label
import matplotlib.pyplot as plt

MODEL_PATH = "/files/private/notebooks/chronos-forecasting/t5_benign_workdir/run-0/checkpoint-final/"
GT_NPZ = "/files/private/notebooks/scripts/aggregated_data.npz"
OUT_DIR = "/files/private/notebooks/results/"
CONTEXT_LEN = 50
TEST_SIZE = 0.3
RANDOM_STATE = 42
MIN_CLUSTER_LEN = 3
DILATION_WIN = 1
CSV_OUT = os.path.join(OUT_DIR, "predictions.csv")
JSON_OUT = os.path.join(OUT_DIR, "metrics.json")
PLOT_PR = os.path.join(OUT_DIR, "pr_curve.png")
PLOT_ROC = os.path.join(OUT_DIR, "roc_curve.png")

os.makedirs(OUT_DIR, exist_ok=True)

#load Chronos model
pipeline = ChronosPipeline.from_pretrained(
    MODEL_PATH, device_map="cpu", torch_dtype=torch.bfloat16
)
data = np.load(GT_NPZ)
series = data["target"]
true_labels = data["labels"].astype(int)
n = len(series)

#extract mean & std embeddings
features = []
labels   = []

for t in range(CONTEXT_LEN, n):
    ctx = torch.tensor(series[t-CONTEXT_LEN:t], dtype=torch.float)
    emb, _ = pipeline.embed(ctx)
    emb = emb.squeeze(0).to(torch.float32)
    mean = emb.mean(dim=0).cpu().numpy()
    std = emb.std(dim=0).cpu().numpy()
    vec = np.concatenate([mean, std], axis=0)
    features.append(vec)
    labels.append(int(true_labels[t]))

X = np.stack(features, axis=0)
y = np.array(labels, dtype=int)
assert X.shape[0] == y.shape[0], f"Mismatch: X rows={X.shape[0]}, y={y.shape[0]}"

#split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)

#train a random forest
clf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train)

#configure threshold
y_score = clf.predict_proba(X_test)[:,1]
cands = np.linspace(y_score.min(), y_score.max(), 200)
best_f1, best_thr = 0.0, cands[0]
for thr in cands:
    y_pred = (y_score >= thr).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

raw_flags = (y_score >= best_thr).astype(int)

#cluster filtering & dilation
labeled, num = label(raw_flags)
cluster_flags = raw_flags.copy()
for region in range(1, num+1):
    if np.sum(labeled==region) < MIN_CLUSTER_LEN:
        cluster_flags[labeled==region] = 0
final_flags = binary_dilation(
    cluster_flags, structure=np.ones(2*DILATION_WIN+1)
).astype(int)
flag_rate = float(final_flags.mean())

#compute & save metrics
precision, recall, _ = precision_recall_curve(y_test, y_score)
pr_auc = float(auc(recall, precision))
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = float(auc(fpr, tpr))
f1_final = float(f1_score(y_test, final_flags))
cm = confusion_matrix(y_test, final_flags).tolist()

metrics = {
    "n_train": len(y_train),
    "n_test": len(y_test),
    "Best_threshold": best_thr,
    "Raw_F1": best_f1,
    "Flag_rate": flag_rate,
    "PR_AUC": pr_auc,
    "ROC_AUC": roc_auc,
    "F1_Score(@best)": f1_final,
    "Confusion_Matrix": cm,
}
with open(JSON_OUT, "w") as fp:
    json.dump(metrics, fp, indent=2)

#save predictions
df = pd.DataFrame({
    "y_true": y_test,
    "y_score": y_score,
    "raw_flag": raw_flags,
    "flag": final_flags,
})
df.to_csv(CSV_OUT, index=False)
