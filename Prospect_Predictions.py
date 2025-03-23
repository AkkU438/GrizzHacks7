import pandas as pd
import numpy as np
import joblib

# Load the model
model_object = joblib.load("ModelSaves/model.joblib")

# Load the data
new_data = pd.read_csv("Data/nfl_draft_prospects_fixed.csv", encoding="Windows-1252")

# Extract player names
player_names = new_data["Name"]  # Using "Name" instead of "Player"

# Ensure categorical columns are treated as strings
for col in model_object["categorical_features"]:
    new_data[col] = new_data[col].astype(str)

# Ensure numerical columns are treated as floats/ints
for col in model_object["numeric_features"]:
    new_data[col] = pd.to_numeric(new_data[col], errors="coerce")

# Drop any columns not present in the training data
expected_features = model_object["numeric_features"] + model_object["categorical_features"]
new_data = new_data[expected_features]

# Make predictions
pred_stage1 = model_object["stage1"].predict(new_data)
pred_stage2 = model_object["stage2"].predict(new_data)

# Combine predictions
raw_predictions = pred_stage1 + pred_stage2

# Print player names with predictions
for name, points in zip(player_names, raw_predictions):
    print(f"{name}: {points:.2f}")
