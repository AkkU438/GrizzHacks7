import pandas as pd
import numpy as np
import joblib

# Load trained model
model_object = joblib.load("ModelSaves/model.joblib")

# Load prospect data
new_data = pd.read_csv("Data/nfl_draft_prospects_fixed.csv", encoding="Windows-1252")
player_names = new_data["Name"]  # Store player names for later

# Ensure data types match the trained model
for col in model_object["categorical_features"]:
    new_data[col] = new_data[col].astype(str)

for col in model_object["numeric_features"]:
    new_data[col] = pd.to_numeric(new_data[col], errors="coerce")

# Ensure same feature order as training
expected_features = model_object["numeric_features"] + model_object["categorical_features"]
new_data = new_data[expected_features]

# Make predictions
pred_stage1 = model_object["stage1"].predict(new_data)
pred_stage2 = model_object["stage2"].predict(new_data)

# Debugging: Print raw predictions before scaling
print("\nRaw Predictions:")
for name, p1, p2 in zip(player_names, pred_stage1, pred_stage2):
    print(f"{name} - Stage 1: {p1:.2f}, Stage 2: {p2:.2f}")

# Apply extreme scaling factor
extreme_scaling_factor = np.where(
    pred_stage1 >= 80, 100.0,
    np.where(pred_stage1 >= 40, 50.0, 10.0)
)
extreme_predictions = pred_stage1 + (extreme_scaling_factor * pred_stage2)

# Print final predictions
print("\nEXTREMELY Conditionally Scaled Predictions:")
for name, points in zip(player_names, extreme_predictions):
    print(f"{name}: {points:.2f}")
