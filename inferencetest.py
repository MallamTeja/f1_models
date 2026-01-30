import time
import requests

URL = "http://localhost:8000/predict"

def predict_position(driver_code, qualifying_time, clean_air_race_pace, rain_prob=0.2, temperature=30):
    payload = {
        "driver_code": driver_code,
        "qualifying_time": qualifying_time,
        "clean_air_race_pace": clean_air_race_pace,
        "rain_prob": rain_prob,
        "temperature": temperature
    }
    try:
        start_time = time.time()
        response = requests.post(URL, json=payload, timeout=15)
        response.raise_for_status()
        end_time = time.time()
        return {
            "result": response.json(),
            "inference_time": end_time - start_time,
            "driver": driver_code
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": str(e), 
            "driver": driver_code,
            "inference_time": 0
        }

def batch_predict(drivers_data, rain_prob=0.2, temperature=30):
    results = []
    for driver in drivers_data:
        result = predict_position(
            driver["driver"],
            driver["qual"],
            driver["pace"],
            rain_prob,
            temperature
        )
        results.append(result)
    return results

rain_prob = 0.2
temperature = 30

drivers_data = [
    {"driver": "RUS", "qual": 82.645, "pace": 91.70},
    {"driver": "VER", "qual": 82.207, "pace": 91.10},
    {"driver": "PIA", "qual": 82.437, "pace": 91.35},
    {"driver": "NOR", "qual": 82.408, "pace": 91.55},
    {"driver": "HAM", "qual": 83.394, "pace": 92.05},
    {"driver": "LEC", "qual": 82.730, "pace": 92.30},
    {"driver": "ALO", "qual": 82.902, "pace": 93.40},
    {"driver": "HUL", "qual": 83.450, "pace": 95.20},
    {"driver": "ALB", "qual": 83.416, "pace": 95.35},
    {"driver": "SAI", "qual": 83.042, "pace": 94.80},
    {"driver": "STR", "qual": 83.097, "pace": 95.10},
    {"driver": "OCO", "qual": 82.913, "pace": 95.50},
    {"driver": "GAS", "qual": 83.468, "pace": 95.55},
]

print("Running API inference test...")
start_time = time.time()
results = batch_predict(drivers_data, rain_prob, temperature)
end_time = time.time()

success = 0
fail = 0
inference_times = []

for result in results:
    if "error" in result:
        fail += 1
    else:
        success += 1
        inference_times.append(result["inference_time"])

total_time = end_time - start_time
avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
min_inference_time = min(inference_times) if inference_times else 0
max_inference_time = max(inference_times) if inference_times else 0

print("\n===== API INFERENCE METRICS =====")
print("Successful:", success)
print("Failed:", fail)
print("Total drivers tested:", len(drivers_data))
print("Total API call time:", f"{total_time:.2f} seconds")
print("Average inference time:", f"{avg_inference_time*1000:.2f} ms")
print("Min inference time:", f"{min_inference_time*1000:.2f} ms")
print("Max inference time:", f"{max_inference_time*1000:.2f} ms")

print("==================================\n")

print("Individual Results:")
for result in results:
    if "error" in result:
        print(f"{result['driver']}: ERROR - {result['error']}")
    else:
        pred_time = result['inference_time'] * 1000
        print(f"{result['driver']}: SUCCESS - {pred_time:.2f}ms")