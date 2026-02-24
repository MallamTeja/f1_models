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
from sklearn_json import to_dict, from_json
import json

load_dotenv()
fastf1.Cache.enable_cache("f1_cache")

OPENWEATHER_API = os.getenv("openweatherapi")

LAT, LON = 30.1328, -97.6411
FORECAST_TIME = "2025-10-19 14:00:00"

session_2024 = fastf1.get_session(2024, "United States", "R")
session_2024.load()

laps_2024 = session_2024.laps[
    ["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
].dropna()

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

sector_times_2024 = laps_2024.groupby("Driver").agg({
    : "mean",
    : "mean",
    : "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

clean_air_race_pace = {
    : 92.10,
    : 93.25,
    : 94.30,
    : 94.40,
    : 94.45,
    : 94.50,
    : 94.60,
    : 95.10,
    : 95.25,
    : 95.60,
    : 95.70,
    : 97.20,
    : 95.30,
    : 95.80,
    : 96.00,
    : 96.10,
    : 95.90,
    : 96.20,
    : 96.30,
    : 96.50,
}

qualifying_2025 = pd.DataFrame({
    : [
        , "NOR", "LEC", "RUS", "HAM",
        , "ANT", "BEA", "SAI", "ALO",
        , "LAW", "TSU", "GAS", "COL",
        , "OCO", "STR", "ALB", "HAD",
    ],
    : [
        92.510,
        92.801,
        92.807,
        92.826,
        92.912,
        93.084,
        93.114,
        93.139,
        93.150,
        93.160,
        93.551,
        93.549,
        93.935,
        93.599,
        94.039,
        94.125,
        94.136,
        94.540,
        94.690,
        999.999,
    ],
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

weather = requests.get(
    
    ,
    timeout=10
).json()

forecast = next(
    (f for f in weather.get("list", []) if f.get("dt_txt") == FORECAST_TIME),
    None
)

rain_probability = forecast.get("pop", 0) if forecast else 0
temperature = forecast.get("main", {}).get("temp", 28) if forecast else 28

team_points = {
    : 0,
    : 0,
    : 0,
    : 0,
    : 0,
    : 0,
    : 0,
    : 0,
    : 0,
    : 0,
}

team_score = {k: (v / max(team_points.values()) if max(team_points.values()) else 0.5)
              for k, v in team_points.items()}

driver_to_team = {
    : "Red Bull",
    : "Red Bull",
    : "McLaren",
    : "McLaren",
    : "Ferrari",
    : "Ferrari",
    : "Mercedes",
    : "Mercedes",
    : "Williams",
    : "Williams",
    : "Haas",
    : "Haas",
    : "Aston Martin",
    : "Aston Martin",
    : "Kick Sauber",
    : "Kick Sauber",
    : "Racing Bulls",
    : "Racing Bulls",
    : "Alpine",
    : "Alpine",
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = (
    qualifying_2025["Team"].map(team_score)
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
        ,
        ,
        ,
        ,
        
    ]
]

y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.1, random_state=39
)

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.9,
    random_state=39,
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

explainer = shap.Explainer(model)
shap_values = explainer(X_train)

shap.summary_plot(
    shap_values,
    X_train,
    feature_names=[
        ,
        ,
        ,
        ,
        
    ],
    show=False
)

plt.tight_layout()
plt.show()

serialized_model = to_dict(model)

artifact = {
    : serialized_model,
    : {
        : imputer.strategy,
        : imputer.statistics_.tolist() if hasattr(imputer, "statistics_") else None
    },
    : [
        ,
        ,
        ,
        ,
        
    ],
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open("us_model.json", "w") as f:
    json.dump(artifact, f, cls=NumpyEncoder)

print("us_model.json saved successfully")
