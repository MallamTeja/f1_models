# 🏎️ F1 Race Pace Predictor


[![Live Demo](https://img.shields.io/badge/Live-f1predictor.tech-FF1801?style=for-the-badge&logo=formula1)](https://f1predictor.tech/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

The **F1 Race Pace Predictor** is a production-grade Machine Learning ecosystem designed to forecast Formula 1 Grand Prix outcomes. By integrating historical racing data from **FastF1**, real-time environmental conditions via **OpenWeather API**, and advanced ensemble models, the system provides high-fidelity predictions for driver lap times across various international circuits.

---

## 🌍 Live Deployment
The project is fully deployed and accessible via a custom domain with modern networking support:
* **Primary URL:** [https://f1predictor.tech](f1predictor.tech)
* **Subdomain Support:** Fully configured for both the root domain (`@`) and `www` subdomain.
* **Networking:** Dual-stack support for both **IPv4** and **IPv6** traffic.
* **Infrastructure:** Hosted on **Render** with automated CI/CD pipelines and SSL encryption.

---

## 🧠 Data Science & Feature Engineering
The core of this project lies in transforming raw telemetry and historical logs into high-impact features for predictive modeling.

### **The Data Pipeline**
* **Historical Extraction:** Utilizes the `FastF1` library to pull session-specific lap times and sector breakdowns from the 2024 season.
* **Real-Time Weather:** Ingests live API data from `OpenWeather` to factor in track temperature and rain probability.
* **Feature Imputation:** Employs `SimpleImputer` with a median strategy to handle missing historical sector data, ensuring model stability during real-time inference.

### **Advanced Feature Engineering**
* **Team Performance Score:** A normalized metric derived by dividing current constructor points by the maximum points in the field, quantifying the "mechanical advantage" of each car.
* **Clean Air Race Pace:** A weighted feature that adjusts expected lap times based on a driver's ability to maintain pace without aerodynamic interference.
* **Total Sector Time:** A composite feature summing the three distinct track sectors to provide a granular view of car-track efficiency.

### **Machine Learning Models**
The project utilizes a diversified ensemble approach to minimize error:
* **Gradient Boosting Regressor (GBR):** Used for the Mexico and US GP models to capture complex, non-linear relationships between qualifying speed and race endurance.
* **XGBoost (XGBRegressor):** Leverages monotone constraints to ensure that as qualifying speed increases, predicted race pace remains logically consistent.
* **Deep Learning:** A Feed-Forward Neural Network (FFN) built with `TensorFlow` for high-dimensional pattern recognition on the Abu Dhabi circuit.

---

## 🚀 Key Features
* **Dynamic Inference Engine:** A **FastAPI** backend that automatically loads specialized `.joblib` model artifacts on startup.
* **Explainable AI (XAI):** Integrated **SHAP** visualizations to provide transparency into how each feature affects the final prediction.
* **Performance Benchmark:** Optimized for high-availability, delivering sub-200ms inference response times even under concurrent load.

---

## 🔧 Setup & Installation

### **1. Clone the Repository**
```bash
git clone [https://github.com/MallamTeja/f1_models.git](https://github.com/MallamTeja/f1_models.git)
cd f1_models
```

2. Local Execution
```
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch API
uvicorn main:app --reload
```
3. Running with Docker

```

# Build the image
docker build -t f1-predictor.

# Run the container
docker run -p 8000:8000 --env-file .env f1-predictor

```
🧪 Testing
Run the automated test suite to verify model integrity and API logic:
```
pytest test_main.py
```
This validates edge cases, including driver code verification and logical lap time ordering—ensuring predicted race pace is realistically slower than qualifying speed.


Disclaimer: This project is unofficial and is not associated in any way with the Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks are trademarks of Formula One Licensing B.V.
