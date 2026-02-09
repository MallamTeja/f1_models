import os
import json
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

LAT, LON = 25.4889, 51.4542
FORECAST_TIME = "2025-11-30 20:00:00"

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
    "VER": 89.10,
    "STR": 95.10,
    "PIA": 91.95,
    "SAI": 95.80,
    "NOR": 91.55,
    "ALO": 93.40,
    "LEC": 93.30,
    "HUL": 95.20,
    "HAM": 96.05,
    "RUS": 95.00,
    "GAS": 95.55,
    "OCO": 95.50,
    "ALB": 95.35,
}

qualifying_2025 = pd.DataFrame(
    {
        "Driver": [
            "RUS",
            "VER",
            "PIA",
            "NOR",
            "HAM",
            "LEC",
            "ALO",
            "HUL",
            "ALB",
            "SAI",
            "STR",
            "OCO",
            "GAS",
        ],
        "QualifyingTime": [
            79.662,
            79.651,
            79.387,
            79.495,
            80.907,
            80.561,
            80.418,
            80.353,
            80.629,
            80.287,
            81.058,
            80.864,
            80.477,
        ],
    }
)


qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(
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
    "McLaren": 800,
    "Mercedes": 459,
    "Red Bull": 426,
    "Ferrari": 383,
    "Williams": 137,
    "Haas": 73,
    "Aston Martin": 80,
    "Kick Sauber": 68,
    "Racing Bulls": 92,
    "Alpine": 22,
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
    "OCO": "Haas",
    "STR": "Aston Martin",
}

team_score = {k: v / max(team_points.values()) for k, v in team_points.items()}

qualifying_2025["TeamPerformanceScore"] = (
    qualifying_2025["Driver"].map(driver_to_team).map(team_score)
)

merged_data = qualifying_2025.merge(
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

print("\nPredicted Qatar 2025 Race Pace – Top 5")
print(top5)

print(
    f"\nMAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f} s"
)

explainer = shap.Explainer(model)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, show=False)

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

def save_model_to_json(model, filepath):
    model_data = {
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "random_state": model.random_state,
        "estimators": []
    }
    
    if hasattr(model, 'estimators_'):
        for stage in model.estimators_:
            stage_trees = []
            for tree_regressor in stage:
                tree = tree_regressor.tree_
                
                def to_list(arr):
                    return arr.tolist() if hasattr(arr, 'tolist') else list(arr)
                
                tree_data = {
                    "children_left": to_list(tree.children_left),
                    "children_right": to_list(tree.children_right),
                    "feature": to_list(tree.feature),
                    "threshold": to_list(tree.threshold),
                    "value": to_list(tree.value), 
                    "node_count": tree.node_count,
                }
                stage_trees.append(tree_data)
            model_data["estimators"].append(stage_trees)
    
    if hasattr(model, 'init_'):
        try:
            dummy_input = np.zeros((1, model.n_features_in_))
            init_val = model.init_.predict(dummy_input)[0]
            if isinstance(init_val, np.ndarray) or isinstance(init_val, list):
                 init_val = init_val[0]
            model_data["init_value"] = float(init_val)
        except Exception as e:
            print(f"Warning: Could not save init value: {e}")
            model_data["init_value"] = 0.0

    with open(filepath, "w") as f:
        json.dump(model_data, f, indent=2)

save_model_to_json(model, "qatarmodel.json")
print("qatarmodel.json saved successfully")