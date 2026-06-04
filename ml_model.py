import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("dataset_air.csv")

target_cols = [
    "pm25",
    "pm10",
    "co2",
    "nox",
    "sox",
    "nhx"
]

#external_cols = [
#   "temperature",
#   "humidity",
#    "wind_speed",
#    "rainfall",
#    "traffic_index"
#]

# Vérification colonnes
required_cols = target_cols #+ external_cols

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise Exception(f"Colonnes manquantes : {missing}")

# =========================
# FEATURES
# =========================

feature_cols = target_cols #+ external_cols

WINDOW = 10

values = df[feature_cols].values

X = []
y = []

for i in range(WINDOW, len(df) - 1):

    # historique complet
    X.append(values[i-WINDOW:i])

    # pollution à t+1
    y.append(df[target_cols].iloc[i+1].values)

X = np.array(X)
y = np.array(y)

print("Shape X :", X.shape)
print("Shape y :", y.shape)

# =========================
# TRAIN / TEST SPLIT
# =========================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================
# FLATTEN POUR XGBOOST
# =========================

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# =========================
# MODEL
# =========================

base_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model = MultiOutputRegressor(base_model)

model.fit(X_train_flat, y_train)

# =========================
# PREDICTION
# =========================
print("X SHAPE =", X.shape)

try:
    print("MODEL FEATURES =", model.estimators_[0].n_features_in_)
except Exception as e:
    print(e)

y_pred = model.predict(X_test_flat)

# =========================
# GLOBAL METRICS
# =========================

print("\n===== GLOBAL METRICS =====")

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# =========================
# METRICS PAR POLLUANT
# =========================

print("\n===== METRICS PAR POLLUANT =====")

for i, col in enumerate(target_cols):

    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2_i = r2_score(y_test[:, i], y_pred[:, i])

    print(f"\n--- {col.upper()} ---")
    print("MAE :", mae_i)
    print("RMSE:", rmse_i)
    print("R2  :", r2_i)

# =========================
# VISUALISATION
# =========================

plt.figure(figsize=(14, 6))

for i in range(2):

    plt.subplot(1, 2, i + 1)

    plt.plot(y_test[:100, i], label="Réel")
    plt.plot(y_pred[:100, i], label="Prédit")

    plt.title(target_cols[i].upper())
    plt.legend()

plt.tight_layout()
plt.show()

# =========================
# DISTRIBUTION ERREURS
# =========================

errors = y_test - y_pred

plt.figure(figsize=(10, 5))
plt.hist(errors.flatten(), bins=50)
plt.title("Distribution des erreurs")
plt.show()

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "air_xgb_model.pkl", compress=3)
joblib.dump(feature_cols, "feature_cols.pkl", compress=3)

print("\nMODEL XGBOOST OK ✔️")