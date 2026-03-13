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
import joblib

load_dotenv()
cache_dir = os.path.join(os.path.dirname(__file__), '..', 'f1_cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

OPENWEATHER_API = os.getenv("OPENWEATHER_API")
LAT, LON = 30.1328, -97.6411
FORECAST_TIME = "2025-10-19 14:00:00"

session_2024 = fastf1.get_session(2024, "United States", "R")
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
    "VER": 92.10,
    "PER": 93.25,
    "NOR": 94.30,
    "PIA": 94.40,
    "LEC": 94.45,
    "RUS": 94.50,
    "SAI": 94.60,
    "HAM": 95.10,
    "LAW": 95.25,
    "ALO": 95.60,
    "TSU": 95.70,
    "MAG": 97.20,
    "ALB": 95.30,
    "HUL": 95.80,
    "OCO": 96.00,
    "GAS": 96.10,
    "BOT": 95.90,
    "ZHO": 96.20,
    "STR": 96.30,
    "COL": 96.50
}

qualifying_2025 = pd.DataFrame({
    "Driver": [
        "VER", "NOR", "LEC", "RUS", "HAM",
        "PIA", "ANT", "BEA", "SAI", "ALO",
        "LAW", "TSU", "GAS", "COL", "OCO",
        "STR", "ALB", "HAD", "BOT", "HUL"
    ],
    "QualifyingTime (s)": [
        92.510, 92.801, 92.807, 92.826, 92.912,
        93.084, 93.114, 93.139, 93.150, 93.160,
        93.551, 93.549, 93.935, 93.599, 94.039,
        94.125, 94.136, 94.540, 94.690, 999.999
    ]
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

weather_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API}&units=metric"
weather = requests.get(weather_url, timeout=10).json()
forecast = next((f for f in weather.get("list", []) if f.get("dt_txt") == FORECAST_TIME), None)
rain_probability = forecast.get("pop", 0) if forecast else 0
temperature = forecast.get("main", {}).get("temp", 28) if forecast else 28

team_points = {
    "Red Bull": 650,
    "McLaren": 620,
    "Ferrari": 580,
    "Mercedes": 420,
    "Williams": 150,
    "Haas": 120,
    "Aston Martin": 100,
    "Kick Sauber": 80,
    "Racing Bulls": 60,
    "Alpine": 40
}

max_points = max(team_points.values()) if team_points else 0
team_score = {k: (v / max_points if max_points else 0.5) for k, v in team_points.items()}

driver_to_team = {
    "VER": "Red Bull",
    "PER": "Red Bull",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "LEC": "Ferrari",
    "SAI": "Ferrari",
    "RUS": "Mercedes",
    "HAM": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "TSU": "Racing Bulls",
    "LAW": "Williams",
    "OCO": "Alpine",
    "GAS": "Alpine",
    "ALB": "Williams",
    "BEA": "Haas",
    "HUL": "Haas",
    "MAG": "Racing Bulls",
    "COL": "Kick Sauber",
    "BOT": "Kick Sauber",
    "HAD": "Williams",
    "ANT": "Kick Sauber"
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
    n_estimators=400,
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

print("\nPredicted US GP Race Pace – Top 5")
print(top5)
print(f"\nMAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s")

# Standardized SHAP Explainer
try:
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    
    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_cols,
        show=False
    )
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/shap_usa.png")
except Exception as e:
    print(f"SHAP error: {e}")

artifact = {
    "model": model,
    "imputer": imputer,
    "features": feature_cols
}

# Save to standardized models directory
joblib.dump(artifact, "models/us_model.joblib")
print("models/us_model.joblib saved successfully")
