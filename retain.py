"""
Run this script once to retrain the model with better hyperparameters.
Place this file in the same folder as your CSV and model.pkl.

Usage:
    python retrain.py
"""

import numpy as np
import pandas as pd
import pickle

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("Smartphone addiction prediction (Responses) - Form Responses 1.csv")

df.columns = [
    "timestamp", "hours_spend_per_day", "check_freq", "class_use",
    "entertainment", "dependency", "check_without_noti", "before_bed", "target"
]

# ── 2. Encode ──────────────────────────────────────────────────────────────────
hours_map          = {"1-2 hours": 1, "3-4 hours": 2, "5 or more hours": 3}
check_map          = {"Rarely or never": 0, "Only when I get notifications": 1,
                      "Every hour": 2, "Every 15-30 minutes": 3, "Every 5 minutes": 4}
class_use_map      = {"Rarely": 0, "Occasionally": 1, "Frequently": 2, "Always": 3}
entertainment_map  = {"Less than 1 hour": 0, "1-2 hours": 1, "3-4 hours": 2, "5 or more hours": 3}
dependency_map     = {"Not at all": 0, "Slightly": 1, "Moderately": 2, "Very much": 3, "Absolutely": 4}
check_noti_map     = {"Never": 0, "Rarely": 1, "Occasionally": 2, "Frequently": 3, "Always": 4}
before_bed_map     = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3, "Always": 4}
target_map         = {"No, I don't": 0, "Yes, I do.": 1}

df["hours_spend_per_day"] = df["hours_spend_per_day"].map(hours_map)
df["check_freq"]          = df["check_freq"].map(check_map)
df["class_use"]           = df["class_use"].map(class_use_map)
df["entertainment"]       = df["entertainment"].map(entertainment_map)
df["dependency"]          = df["dependency"].map(dependency_map)
df["check_without_noti"]  = df["check_without_noti"].map(check_noti_map)
df["before_bed"]          = df["before_bed"].map(before_bed_map)
df["target"]              = df["target"].map(target_map)
df = df.dropna().reset_index(drop=True)

# ── 3. Features & target ───────────────────────────────────────────────────────
feature_cols = [
    "hours_spend_per_day", "class_use", "entertainment",
    "dependency", "check_without_noti", "before_bed"
]
X = df[feature_cols].values.astype(float)
y = df["target"].values.astype(int).reshape(-1, 1)

# ── 4. Train/test split ────────────────────────────────────────────────────────
np.random.seed(42)
indices    = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]

# ── 5. Scale ───────────────────────────────────────────────────────────────────
x_mean = np.mean(X_train, axis=0)
x_std  = np.std(X_train, axis=0)
x_std[x_std == 0] = 1

X_train_s = (X_train - x_mean) / x_std
X_test_s  = (X_test  - x_mean) / x_std

X_train_b = np.concatenate((np.ones((X_train_s.shape[0], 1)), X_train_s), axis=1)
X_test_b  = np.concatenate((np.ones((X_test_s.shape[0],  1)), X_test_s),  axis=1)

# ── 6. Logistic regression functions ──────────────────────────────────────────
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def gradient(X, y, theta):
    m = len(y)
    h = sigmoid(np.matmul(X, theta))
    return (1 / m) * np.matmul(X.T, (h - y))

# ── 7. Train with better hyperparameters ───────────────────────────────────────
#   ↓ lr 0.1 → 0.01  +  iterations 100000 → 5000
#   This prevents theta from blowing up while still converging
theta      = np.zeros((X_train_b.shape[1], 1))
lr         = 0.01       # was 0.1  ← main fix
iterations = 5000       # was 100000

for i in range(iterations):
    theta -= lr * gradient(X_train_b, y_train, theta)

# ── 8. Evaluate ────────────────────────────────────────────────────────────────
test_probs = sigmoid(np.matmul(X_test_b, theta))
test_pred  = (test_probs >= 0.5).astype(int)
acc        = np.mean(test_pred == y_test)

print("Final theta:\n", theta)
print("\nx_mean:", x_mean)
print("x_std: ", x_std)
print(f"\nTest Accuracy: {acc*100:.1f}%")

# Quick sanity check — should now give values spread between 0 and 1
for label, raw in [
    ("Low risk  (1,0,0,0,0,0)", [[1, 0, 0, 0, 0, 0]]),
    ("Mid risk  (2,1,1,2,2,2)", [[2, 1, 1, 2, 2, 2]]),
    ("High risk (3,3,3,4,4,4)", [[3, 3, 3, 4, 4, 4]]),
]:
    r  = np.array(raw, dtype=float)
    sc = (r - x_mean) / x_std
    Xn = np.concatenate((np.ones((1, 1)), sc), axis=1)
    p  = float(sigmoid(np.matmul(Xn, theta)).item())
    print(f"  {label}  →  prob = {p*100:.1f}%")

# ── 9. Save ────────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"theta": theta, "x_mean": x_mean, "x_std": x_std}, f)

print("\n✅ model.pkl saved with well-calibrated theta values.")