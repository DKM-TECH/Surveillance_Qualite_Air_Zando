import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")
df = df.dropna()

# =========================
# FEATURES
# =========================
features = ["pm25","pm10","co2","nox","sox","nhx"]
X = df[features]

# =========================
# TARGETS FUTURES
# =========================
y = df[
    ["pm25_future","pm10_future","co2_future",
     "nox_future","sox_future","nhx_future"]
]

# =========================
# TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("MAE =", mae)
print("RMSE =", rmse)
print("R2 =", r2)

print("MAE =", mae)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "air_model.pkl")

print("✅ MODEL ENTRAÎNÉ ET SAUVEGARDÉ")