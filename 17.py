import os
import fastf1
import pandas as pd
import numpy as np
import requests
import shap

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

load_dotenv()
fastf1.Cache.enable_cache("f1_cache")

OPENWEATHER_API = os.getenv("openweatherapi")
if not OPENWEATHER_API:
    raise RuntimeError("openweatherapi not set")

LAT, LON = 24.4672, 54.6031
FORECAST_TIME = "2025-12-07 13:00:00"

session = fastf1.get_session(2024, 24, "R")
session.load()

laps = session.laps[
    ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
].dropna()

for c in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps[f"{c} (s)"] = laps[c].dt.total_seconds()

sector = laps.groupby("Driver").agg(
    {
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean"
    }
).reset_index()

sector["TotalSectorTime (s)"] = (
    sector["Sector1Time (s)"]
    + sector["Sector2Time (s)"]
    + sector["Sector3Time (s)"]
)

clean_air_race_pace = {
    "VER": 92.95,
    "PIA": 93.05,
    "NOR": 93.22,
    "LEC": 93.40,
    "RUS": 93.83,
    "HAM": 94.02,
    "ALO": 94.78,
    "SAI": 94.50,
    "STR": 95.32,
    "HUL": 95.35,
    "OCO": 95.68,
    "ALB": 95.50,
    "GAS": 95.60
}

qualifying = pd.DataFrame({
    "Driver": list(clean_air_race_pace.keys()),
    "QualifyingTime": [
        82.207,
        82.437,
        82.408,
        82.730,
        82.645,
        83.394,
        82.902,
        83.042,
        83.097,
        83.450,
        82.913,
        83.416,
        83.468
    ]
})

qualifying["CleanAirRacePace (s)"] = qualifying["Driver"].map(clean_air_race_pace)

merged = qualifying.merge(
    sector[["Driver", "TotalSectorTime (s)"]],
    on="Driver",
    how="left"
)

weather_url = (
    f"http://api.openweathermap.org/data/2.5/forecast"
    f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API}&units=metric"
)

weather_data = requests.get(weather_url, timeout=10).json()

forecast = next(
    (f for f in weather_data.get("list", [])
     if f.get("dt_txt") == FORECAST_TIME),
    None
)

merged["RainProbability"] = forecast.get("pop", 0) if forecast else 0
merged["Temperature"] = forecast.get("main", {}).get("temp", 20) if forecast else 20

team_points = {
    "McLaren": 800,
    "Mercedes": 459,
    "Red Bull": 426,
    "Williams": 137,
    "Ferrari": 382,
    "Haas": 73,
    "Aston Martin": 80,
    "Kick Sauber": 68,
    "Alpine": 22
}

driver_to_team = {
    "VER": "Red Bull",
    "PIA": "McLaren",
    "NOR": "McLaren",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "RUS": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "ALB": "Williams",
    "SAI": "Williams",
    "OCO": "Haas",
    "HUL": "Kick Sauber",
    "GAS": "Alpine"
}

max_pts = max(team_points.values())
team_score = {k: v / max_pts for k, v in team_points.items()}

merged["TeamPerformanceScore"] = (
    merged["Driver"]
    .map(driver_to_team)
    .map(team_score)
)

y_time = laps.groupby("Driver")["LapTime (s)"].mean()

data = merged.copy()
data["y"] = data["Driver"].map(y_time)
data = data.dropna(subset=["y"])

X = data[
    [
        "QualifyingTime",
        "CleanAirRacePace (s)",
        "TotalSectorTime (s)",
        "TeamPerformanceScore",
        "RainProbability",
        "Temperature"
    ]
]

y = data["y"]

X = SimpleImputer(strategy="median").fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=39
)

model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
    
])

model.compile(      
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss="mae"
)

model.fit(
    X_train,
    y_train,
    epochs=400,
    batch_size=8,
    verbose=0,
    callbacks=[
        callbacks.EarlyStopping(
            patience=40,
            restore_best_weights=True
        )
    ]
)

data["PredictedRaceTime (s)"] = model.predict(X).ravel()

final = data.sort_values(
    ["PredictedRaceTime (s)", "QualifyingTime"]
).reset_index(drop=True)

print("\nPredicted Top 5 (FFN)")
for i in range(5):
    print(f"P{i+1}: {final.iloc[i]['Driver']}")

y_pred = model.predict(X_test).ravel()
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")



model.save("abudhabiffnmodel.keras")
