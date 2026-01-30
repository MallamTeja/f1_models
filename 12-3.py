import os
import fastf1
import pandas as pd
import numpy as np
import requests
import joblib
import shap
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


load_dotenv()
fastf1.Cache.enable_cache("f1_cache")



OPENWEATHER_API = os.getenv("openweatherapi")
LAT, LON = 24.4672, 54.6031
FORECAST_TIME = "2025-12-07 13:00:00"

session_2024 = fastf1.get_session(2024, 24, "R")
session_2024.load()

laps_2024 = session_2024.laps[
    ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
].dropna()

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"]
    + sector_times_2024["Sector2Time (s)"]
    + sector_times_2024["Sector3Time (s)"]
)

clean_air_race_pace = {
    "VER": 91.10, 
    "HAM": 92.05, 
    "NOR": 91.55,
    "OCO": 95.50,
    "PIA": 91.35,
    "STR": 95.10,
    "ALO": 93.40, 
    "LEC": 92.30,
    "SAI": 94.80, 
    "HUL": 95.20, 
    "RUS": 91.70,
    "ALB": 95.35, 
    "GAS": 95.55
}

qualifying_2025 = pd.DataFrame({
    "Driver": ["RUS", "VER", "PIA", "NOR", "HAM", "LEC", "ALO", "HUL", "ALB", "SAI", "STR", "OCO", "GAS"],
    "QualifyingTime": [
        82.645, 
        82.207, 
        82.437, 
        82.408, 
        83.394,
        82.730, 
        82.902, 
        83.450, 
        83.416,
        83.042, 
        83.097, 
        82.913, 
        83.468
    ]
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

weather = requests.get(
    f"http://api.openweathermap.org/data/2.5/forecast"
    f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API}&units=metric",
    timeout=10
).json()

forecast = next(
    (f for f in weather.get("list", []) if f.get("dt_txt") == FORECAST_TIME),
    None
)

rain_probability = forecast.get("pop", 0) if forecast else 0
temperature = forecast.get("main", {}).get("temp", 20) if forecast else 20

team_points = {
    "McLaren": 800,
    "Mercedes": 459,
    "Red Bull": 426,
    "Williams": 137,
    "Ferrari": 382,
    "Haas": 73,
    "Aston Martin": 80,
    "Kick Sauber": 68,
    "Racing Bulls": 92,
    "Alpine": 22
}

driver_to_team = {
    "VER": "Red Bull",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "LEC": "Ferrari",
    "RUS": "Mercedes",
    "HAM": "Ferrari",
    "GAS": "Alpine",
    "ALO": "Aston Martin",
    "SAI": "Williams",
    "HUL": "Kick Sauber",
    "OCO": "Alpine",
    "STR": "Aston Martin"
}

team_score = {k: v / max(team_points.values()) for k, v in team_points.items()}

qualifying_2025["TeamPerformanceScore"] = (
    qualifying_2025["Driver"]
    .map(driver_to_team)
    .map(team_score)
)

merged_data = qualifying_2025.merge(
    sector_times_2024[["Driver", "TotalSectorTime (s)"]],
    on="Driver",
    how="left"
)

merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

merged_data = merged_data[
    merged_data["Driver"].isin(laps_2024["Driver"].unique())
]

X = merged_data[
    [
        "QualifyingTime",
        "RainProbability",
        "Temperature",
        "TeamPerformanceScore",
        "CleanAirRacePace (s)"
    ]
]

y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

X = SimpleImputer(strategy="median").fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=39
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.9,
    max_depth=3,
    random_state=39,
    monotone_constraints="(1, 0, 0, -1, -1)"
)

model.fit(X_train, y_train)

merged_data["PredictedLapTime (s)"] = model.predict(X)

top5 = (
    merged_data
    .sort_values("PredictedLapTime (s)")
    .reset_index(drop=True)
    .loc[:4, ["Driver", "PredictedLapTime (s)"]]
)

top5.index = range(1, 6)

print("\nPredicted Abu Dhabi Race Pace – Top 5")
print(top5)

print(f"\nMAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

explainer = shap.Explainer(model)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.show()



explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

shap.summary_plot(
    shap_values,
    X_train,
    feature_names=[
        "QualifyingTime",
        "RainProbability",
        "Temperature",
        "TeamPerformanceScore",
        "CleanAirRacePace (s)"
    ],
    show=False
)

plt.tight_layout()
plt.show()


model.get_booster().save_model("abudhabimodel.json")
print("abudhabimodel.json saved successfully")