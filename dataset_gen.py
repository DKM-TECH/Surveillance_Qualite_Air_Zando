import pandas as pd
import numpy as np

# =========================
# SIMULATION / BASE DATA
# =========================

np.random.seed(42)

n = 200  # nombre de lignes

data = {
    "pm25": np.linspace(10, 80, n) + np.random.normal(0, 2, n),
    "pm10": np.linspace(20, 120, n) + np.random.normal(0, 3, n),
    "co2": np.linspace(400, 900, n) + np.random.normal(0, 10, n),
    "nox": np.linspace(10, 80, n) + np.random.normal(0, 2, n),
    "sox": np.linspace(5, 40, n) + np.random.normal(0, 1, n),
    "nhx": np.linspace(2, 25, n) + np.random.normal(0, 1, n),
}

df = pd.DataFrame(data)

cols = ["pm25","pm10","co2","nox","sox","nhx"]

WINDOW = 5

X_data = []
y_data = []

# =========================
# SLIDING WINDOW
# =========================

for i in range(WINDOW, len(df)):

    # INPUT (passé)
    window = df[cols].iloc[i-WINDOW:i].values.flatten()

    # OUTPUT (instant t)
    target = df[cols].iloc[i].values

    X_data.append(window)
    y_data.append(target)

# =========================
# DATAFRAME FINAL
# =========================

X_df = pd.DataFrame(X_data)
y_df = pd.DataFrame(y_data)

# nom des colonnes input
X_df.columns = [
    f"{c}_t-{i}"
    for i in range(WINDOW, 0, -1)
    for c in cols
]

y_df.columns = cols

dataset = pd.concat([X_df, y_df], axis=1)

# =========================
# SAVE
# =========================

dataset.to_csv("dataset_timeseries.csv", index=False)

print("DATASET TIMESERIES CRÉÉ ✔️")
print(dataset.head())
print("Shape:", dataset.shape)