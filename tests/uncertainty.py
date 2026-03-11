import xgboost as xgb
import numpy as np
import json
import os

def test_model_uncertainty():
    model_path = "abu_dhabi_model.json"
    lookup_path = "lookup_data.json"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    model = xgb.Booster()
    model.load_model(model_path)
    with open(lookup_path, 'r') as f:
        drivers = json.load(f).get("drivers", {})
    base_features = np.array([[82.207, 0.0, 25.0, drivers.get("VER", 0.5), 91.10]])
    print(f"Testing Uncertainty for Driver: VER")
    print("-" * 40)
    n_iterations = 100
    predictions = []
    for _ in range(n_iterations):
        noise = np.random.normal(0, [0.1, 0.02, 1.0, 0.0, 0.05], base_features.shape)
        noisy_input = base_features + noise
        dmatrix = xgb.DMatrix(noisy_input)
        preds = model.predict(dmatrix)
        predictions.append(preds[0])
    mean_pace = np.mean(predictions)
    std_dev = np.std(predictions)
    ci_95 = (np.percentile(predictions, 2.5), np.percentile(predictions, 97.5))
    print(f"Mean Predicted Pace: {mean_pace:.4f}s")
    print(f"Uncertainty (Std Dev): {std_dev:.4f}s")
    print(f"95% Confidence Interval: [{ci_95[0]:.4f}s, {ci_95[1]:.4f}s]")
    if std_dev < 1.0:
        print("\nResult: LOW UNCERTAINTY")
    elif std_dev < 2.0:
        print("\nResult: MODERATE UNCERTAINTY")
    else:
        print("\nResult: HIGH UNCERTAINTY")
    assert (ci_95[1] - ci_95[0]) < 3.0

if __name__ == "__main__":
    test_model_uncertainty()
