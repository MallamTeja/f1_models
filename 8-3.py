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
from sklearn.ensemble import GradientBoostingRegressor

load_dotenv()
fastf1.Cache.enable_cache("f1_cache")

OPENWEATHER_API = os.getenv("openweatherapi")

LAT, LON = 36.1147, -115.1728
FORECAST_TIME = "2025-11-23 20:00:00"

session_2024 = fastf1.get_session(2024, 22, "R")
session_2024.load()

laps_2024 = session_2024.laps[
    ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
].dropna()

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

sector_times_2024 = (
    laps_2024.groupby("Driver")
    .agg(
        {
            "Sector1Time (s)": "mean",
            "Sector2Time (s)": "mean",
            "Sector3Time (s)": "mean",
        }
    )
    .reset_index()
)

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"]
    + sector_times_2024["Sector2Time (s)"]
    + sector_times_2024["Sector3Time (s)"]
)

clean_air_race_pace = {
    "PIA": 99.10,
    "VER": 97.35,
    "NOR": 99.10,
    "LAW": 102.40,
    "LEC": 99.20,
    "BOT": 103.10,
    "ALB": 104.50,
    "HUL": 100.50,
    "HAM": 98.80,
    "ZHO": 101.70,
    "TSU": 100.80,
    "PER": 100.90,
    "ALO": 101.30,
    "MAG": 101.40,
    "RUS": 99.80,
    "COL": 101.80,
    "STR": 102.30,
    "OCO": 102.80,
    "SAI": 99.05,
    "GAS": 105.00,
}

qualifying_2024 = pd.DataFrame(
    {
        "Driver": [
            "RUS",
            "SAI",
            "GAS",
            "LEC",
            "VER",
            "NOR",
            "TSU",
            "PIA",
            "HUL",
            "HAM",
            "OCO",
            "MAG",
            "ZHO",
            "COL",
            "LAW",
            "PER",
            "ALO",
            "ALB",
            "BOT",
            "STR",
        ],
        "QualifyingTime": [
            92.312,
            92.410,
            92.664,
            92.783,
            92.797,
            93.008,
            93.029,
            93.033,
            93.062,
            108.106,
            93.221,
            93.297,
            93.566,
            93.749,
            94.257,
            94.155,
            94.258,
            94.425,
            94.430,
            94.484,
        ],
    }
)

qualifying_2024["CleanAirRacePace (s)"] = qualifying_2024["Driver"].map(
    clean_air_race_pace
)

weather = requests.get(
    "http://api.openweathermap.org/data/2.5/forecast"
    f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API}&units=metric",
    timeout=10,
).json()

forecast = next(
    (f for f in weather.get("list", []) if f.get("dt_txt") == FORECAST_TIME),
    None,
)

rain_probability = forecast.get("pop", 0) if forecast else 0
temperature = forecast.get("main", {}).get("temp", 20) if forecast else 20

team_points = {
    "McLaren": 608,
    "Ferrari": 584,
    "Red Bull": 555,
    "Mercedes": 425,
    "Aston Martin": 86,
    "Haas": 50,
    "Alpine": 49,
    "RB": 46,
    "Williams": 17,
    "Kick Sauber": 0,
}

driver_to_team = {
    "VER": "Red Bull",
    "PER": "Red Bull",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "LEC": "Ferrari",
    "SAI": "Ferrari",
    "RUS": "Mercedes",
    "HAM": "Mercedes",
    "GAS": "Alpine",
    "OCO": "Alpine",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "HUL": "Haas",
    "MAG": "Haas",
    "TSU": "RB",
    "LAW": "RB",
    "ALB": "Williams",
    "COL": "Williams",
    "BOT": "Kick Sauber",
    "ZHO": "Kick Sauber",
}

team_score = {k: v / max(team_points.values()) for k, v in team_points.items()}

qualifying_2024["TeamPerformanceScore"] = (
    qualifying_2024["Driver"].map(driver_to_team).map(team_score)
)

merged_data = qualifying_2024.merge(
    sector_times_2024[["Driver", "TotalSectorTime (s)"]],
    on="Driver",
    how="left",
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
        "CleanAirRacePace (s)",
    ]
]

y = (
    laps_2024.groupby("Driver")["LapTime (s)"]
    .mean()
    .reindex(merged_data["Driver"])
)

X = SimpleImputer(strategy="median").fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=39
)

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=39,
)

model.fit(X_train, y_train)

merged_data["PredictedLapTime (s)"] = model.predict(X)

top5 = (
    merged_data.sort_values("PredictedLapTime (s)")
    .reset_index(drop=True)
    .loc[:4, ["Driver", "PredictedLapTime (s)"]]
)
top5.index = range(1, 6)

print("\nPredicted Las Vegas 2024 Race Pace – Top 5")
print(top5)

print(
    f"\nMAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s"
)

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
        "CleanAirRacePace (s)",
    ],
    show=False,
)
plt.tight_layout()
plt.show()


model.get_booster().save_model("lasvegasmodel.json")
print("lasvegasmodel.json saved successfully")
