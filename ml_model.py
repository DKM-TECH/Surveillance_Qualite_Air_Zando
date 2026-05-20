import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Charger dataset
df = pd.read_csv("dataset.csv")

X = df[[
    "pm25",
    "pm10",
    "co2",
    "nox",
    "sox",
    "nhx"
]]

y = df[[
    "pm25_future",
    "pm10_future",
    "co2_future",
    "nox_future",
    "sox_future",
    "nhx_future"
]]

# séparation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# modèle
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# entraînement
model.fit(X_train, y_train)

# prédiction
pred = model.predict(X_test)

# erreur
mae = mean_absolute_error(y_test, pred)

print("MAE =", mae)

# sauvegarde
joblib.dump(model, "air_model.pkl")

print("MODELE SAUVEGARDE")