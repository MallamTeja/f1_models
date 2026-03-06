import os
import fastf1
import pandas as pd
import numpy as np
import requests
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn_json import to_dict
import json

load_dotenv()
fastf1.Cache.enable_cache("f1_cache")

OPENWEATHER_API = os.getenv("openweatherapi") or os.getenv("OPENWEATHER_API")

# Autódromo Hermanos Rodríguez, Mexico City
LAT, LON = 19.4042, -99.0907
FORECAST_TIME = "2025-10-26 14:00:00"

session_2024 = fastf1.get_session(2024, "Mexico", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna()

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

clean_air_race_pace = {
    "VER": 80.50,
    "PER": 81.20,
    "NOR": 80.80,
    "PIA": 81.00,
    "LEC": 80.70,
    "SAI": 80.60,
    "RUS": 81.30,
    "HAM": 81.40,
    "ALO": 81.80,
    "STR": 82.20,
    "TSU": 82.50,
    "LAW": 82.60,
    "ALB": 82.40,
    "MAG": 82.80,
    "HUL": 82.70,
    "GAS": 82.90,
    "OCO": 83.00,
    "BOT": 83.20,
    "ZHO": 83.40,
    "COL": 83.50
}

qualifying_2025 = pd.DataFrame({
    "Driver": list(clean_air_race_pace.keys()),
    "QualifyingTime (s)": [
        77.100, 77.600, 77.200, 77.400, 77.300, 77.250, 77.500, 77.700, 78.100, 78.400,
        78.600, 78.800, 78.500, 79.100, 79.000, 79.300, 79.400, 79.600, 79.800, 80.000
    ]
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

rain_probability = 0
temperature = 22

if OPENWEATHER_API:
    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API}&units=metric"
        weather_resp = requests.get(weather_url, timeout=10)
        if weather_resp.status_code == 200:
            weather = weather_resp.json()
            forecast = next((f for f in weather.get("list", []) if f.get("dt_txt") == FORECAST_TIME), None)
            if forecast:
                rain_probability = forecast.get("pop", 0)
                temperature = forecast.get("main", {}).get("temp", 22)
    except Exception as e:
        pass

team_points = {
    "Red Bull": 650, "McLaren": 620, "Ferrari": 580, "Mercedes": 420, "Williams": 150,
    "Haas": 120, "Aston Martin": 100, "Kick Sauber": 80, "Racing Bulls": 60, "Alpine": 40
}

max_points = max(team_points.values())
team_score = {k: v / max_points for k, v in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "PER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren",
    "LEC": "Ferrari", "SAI": "Ferrari", "RUS": "Mercedes", "HAM": "Mercedes",
    "ALO": "Aston Martin", "STR": "Aston Martin", "TSU": "Racing Bulls",
    "LAW": "Williams", "OCO": "Alpine", "GAS": "Alpine", "ALB": "Williams",
    "MAG": "Haas", "HUL": "Haas", "COL": "Kick Sauber", "BOT": "Kick Sauber", "ZHO": "Kick Sauber"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_score)

merged_data = qualifying_2025.merge(
    sector_times_2024[["Driver", "TotalSectorTime (s)"]],
    on="Driver",
    how="left"
)

merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data = merged_data[merged_data["Driver"].isin(laps_2024["Driver"].unique())]

feature_cols = [
    "QualifyingTime (s)",
    "CleanAirRacePace (s)",
    "TeamPerformanceScore",
    "TotalSectorTime (s)",
    "RainProbability"
]

X = merged_data[feature_cols]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.1, random_state=39
)

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=39
)

model.fit(X_train, y_train)

merged_data["PredictedLapTime (s)"] = model.predict(X_imputed)
top5 = (
    merged_data.sort_values("PredictedLapTime (s)")
    .reset_index(drop=True)
    .loc[:4, ["Driver", "PredictedLapTime (s)"]]
)
top5.index = range(1, 6)

print("\nPredicted Mexico GP Race Pace – Top 5")
print(top5)
if not y_test.empty:
    print(f"\nMAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

explainer = shap.Explainer(model)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig("shap_mexico.png")

artifact = {
    "model": to_dict(model),
    "imputer": {
        "strategy": imputer.strategy,
        "statistics": imputer.statistics_.tolist()
    },
    "features": feature_cols
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open("mexico_model.json", "w") as f:
    json.dump(artifact, f, cls=NumpyEncoder)

print("mexico_model.json saved successfully")
