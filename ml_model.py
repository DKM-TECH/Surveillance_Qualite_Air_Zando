import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import joblib

# =========================
# CHARGER DATASET
# =========================

df = pd.read_csv("dataset.csv")

# suppression valeurs nulles
df = df.dropna()

# =========================
# FEATURES
# =========================

X = df[[
    "pm25",
    "pm10",
    "co2",
    "nox",
    "sox",
    "nhx"
]]

# =========================
# TARGETS FUTURES
# =========================

y = df[[
    "pm25_future",
    "pm10_future",
    "co2_future",
    "nox_future",
    "sox_future",
    "nhx_future"
]]

# =========================
# TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODELE IA
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# entraînement
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, pred)

print("MAE =", round(mae, 4))

# =========================
# SAUVEGARDE
# =========================

joblib.dump(model, "air_model.pkl")

print("MODELE IA SAUVEGARDE")