import requests
import time
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

requests_payloads = []

for _ in range(REPEAT_PER_DRIVER):
    for d in drivers_data:
        requests_payloads.append({
            "driver_code": d["driver"],
            "qualifying_time": d["qual"],
            "clean_air_race_pace": d["pace"],
            "rain_prob": rain_prob,
            "temperature": temperature
        })

TOTAL_REQUESTS = len(requests_payloads)


def send_request(payload):
    r = requests.post(URL, json=payload, timeout=15)
    return r.status_code


start_time = time.time()

success = 0
fail = 0

with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:

    futures = [
        executor.submit(send_request, payload)
        for payload in requests_payloads
    ]

    for future in as_completed(futures):
        try:
            status = future.result()
            if status == 200:
                success += 1
            else:
                fail += 1
        except Exception:
            fail += 1


end_time = time.time()

total_time = end_time - start_time

throughput = TOTAL_REQUESTS / total_time


print("\n===== DRIVER LOAD TEST RESULTS =====")
print("Drivers tested:", len(drivers_data))
print("Total Requests:", TOTAL_REQUESTS)
print("Concurrent Users:", CONCURRENT_USERS)
print("Successful:", success)
print("Failed:", fail)
print("Total Time:", round(total_time, 2), "seconds")
print("Throughput:", round(throughput, 2), "req/sec")
print("===================================\n")
