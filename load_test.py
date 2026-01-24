import requests
import time
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

print("LOAD TEST STARTED")

URL = "https://r14-k1ow.onrender.com/predict"

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

CONCURRENT_USERS = 10
REPEAT_PER_DRIVER = 5

payloads = []

for _ in range(REPEAT_PER_DRIVER):
    for d in drivers_data:
        payloads.append({
            "driver_code": d["driver"],
            "qualifying_time": d["qual"],
            "clean_air_race_pace": d["pace"],
            "rain_prob": rain_prob,
            "temperature": temperature
        })

TOTAL_REQUESTS = len(payloads)

results = []


def send_request(payload):

    start = time.time()

    try:
        r = requests.post(URL, json=payload, timeout=15)
        latency = time.time() - start

        if r.status_code == 200:
            data = r.json()

            return {
                "driver": payload["driver_code"],
                "qualifying_time": payload["qualifying_time"],
                "clean_air_race_pace": payload["clean_air_race_pace"],
                "rain_prob": payload["rain_prob"],
                "temperature": payload["temperature"],
                "predicted_pace": data.get("predicted_pace"),
                "api_latency": latency,
                "status": "success"
            }

        else:
            return {
                "driver": payload["driver_code"],
                "status": "failed",
                "http_code": r.status_code
            }

    except Exception as e:
        return {
            "driver": payload["driver_code"],
            "status": "error",
            "error": str(e)
        }


start_time = time.time()

with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:

    futures = [executor.submit(send_request, p) for p in payloads]

    for future in as_completed(futures):
        result = future.result()
        results.append(result)

        if result["status"] == "success":
            print(
                f"{result['driver']} -> Predicted Pace: {round(result['predicted_pace'],3)} | Latency: {round(result['api_latency'],4)}s"
            )
        else:
            print(f"{result['driver']} -> FAILED")


end_time = time.time()

total_time = end_time - start_time

success_count = sum(1 for r in results if r["status"] == "success")
fail_count = TOTAL_REQUESTS - success_count

throughput = TOTAL_REQUESTS / total_time


print("\n===== DRIVER LOAD TEST RESULTS =====")
print("Drivers:", len(drivers_data))
print("Total Requests:", TOTAL_REQUESTS)
print("Concurrent Users:", CONCURRENT_USERS)
print("Successful:", success_count)
print("Failed:", fail_count)
print("Total Time:", round(total_time, 2), "seconds")
print("Throughput:", round(throughput, 2), "req/sec")
print("===================================\n")


with open("predictions.json", "w") as jf:
    json.dump(results, jf, indent=4)


with open("predictions.csv", "w", newline="") as cf:

    fieldnames = [
        "driver",
        "qualifying_time",
        "clean_air_race_pace",
        "rain_prob",
        "temperature",
        "predicted_pace",
        "api_latency",
        "status"
    ]

    writer = csv.DictWriter(cf, fieldnames=fieldnames)
    writer.writeheader()

    for row in results:
        if row["status"] == "success":
            writer.writerow(row)


print("Saved output files:")
print("predictions.json")
print("predictions.csv")
