import numpy as np
import pandas as pd

np.random.seed(42)

# =========================
# 5 MOIS DE DONNEES HORAIRES
# =========================

n = 24 * 30 * 5  # 3600 heures

dates = pd.date_range(
    start="2025-01-01",
    periods=n,
    freq="h"
)

df = pd.DataFrame()
df["timestamp"] = dates

# =========================
# VARIABLES EXTERNES
# =========================

hours = df["timestamp"].dt.hour
days = np.arange(n)

# température
temperature = (
    27
    + 4*np.sin(2*np.pi*hours/24)
    + np.random.normal(0,1,n)
)

# humidité
humidity = (
    70
    - 0.8*(temperature-27)
    + np.random.normal(0,3,n)
)

humidity = np.clip(humidity,40,95)

# vitesse vent
wind_speed = (
    6
    + 2*np.sin(2*np.pi*days/200)
    + np.random.normal(0,1,n)
)

wind_speed = np.clip(wind_speed,0,15)

# pluie
rainfall = np.random.binomial(1,0.08,n)
rainfall = rainfall * np.random.uniform(2,25,n)

# trafic
traffic_index = (
    50
    + 40*np.sin((hours-7)/24*2*np.pi)
    + 35*np.sin((hours-18)/24*2*np.pi)
    + np.random.normal(0,5,n)
)

traffic_index = np.clip(traffic_index,0,100)

# =========================
# POLLUANTS
# =========================

pm25 = np.zeros(n)
pm10 = np.zeros(n)
co2 = np.zeros(n)
nox = np.zeros(n)
sox = np.zeros(n)
nhx = np.zeros(n)

pm25[0] = 25
pm10[0] = 40
co2[0] = 500
nox[0] = 20
sox[0] = 8
nhx[0] = 5

for t in range(1,n):

    pm25[t] = (
        0.88*pm25[t-1]
        + 0.18*traffic_index[t]
        - 0.70*wind_speed[t]
        - 0.40*rainfall[t]
        + 0.10*humidity[t]
        + np.random.normal(0,2)
    )

    pm10[t] = (
        0.87*pm10[t-1]
        + 0.22*traffic_index[t]
        - 0.60*wind_speed[t]
        - 0.30*rainfall[t]
        + np.random.normal(0,3)
    )

    co2[t] = (
        0.92*co2[t-1]
        + 1.2*traffic_index[t]
        - 0.5*wind_speed[t]
        + np.random.normal(0,10)
    )

    nox[t] = (
        0.90*nox[t-1]
        + 0.15*traffic_index[t]
        - 0.25*wind_speed[t]
        + np.random.normal(0,1.5)
    )

    sox[t] = (
        0.88*sox[t-1]
        + 0.08*traffic_index[t]
        - 0.12*wind_speed[t]
        + np.random.normal(0,0.7)
    )

    nhx[t] = (
        0.85*nhx[t-1]
        + 0.05*humidity[t]
        - 0.08*rainfall[t]
        + np.random.normal(0,0.5)
    )

# =========================
# BORNES REALISTES
# =========================

pm25 = np.clip(pm25,5,250)
pm10 = np.clip(pm10,10,400)
co2 = np.clip(co2,350,3000)
nox = np.clip(nox,1,200)
sox = np.clip(sox,1,100)
nhx = np.clip(nhx,1,60)

# =========================
# DATAFRAME FINAL
# =========================

df["temperature"] = temperature
df["humidity"] = humidity
df["wind_speed"] = wind_speed
df["rainfall"] = rainfall
df["traffic_index"] = traffic_index

df["pm25"] = pm25
df["pm10"] = pm10
df["co2"] = co2
df["nox"] = nox
df["sox"] = sox
df["nhx"] = nhx

# =========================
# SAVE
# =========================

df.to_csv("dataset_air.csv", index=False)

print("DATASET REALISTE GENERE ✔")
print(df.head())
print(df.shape)