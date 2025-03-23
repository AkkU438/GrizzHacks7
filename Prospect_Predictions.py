import pandas as pd
import numpy as np
import joblib


model_object = joblib.load("ModelSaves/model.joblib")


new_data = pd.read_csv("Data/nfl_draft_prospects_fixed.csv", encoding="Windows-1252")


player_names = new_data["Name"]  


for col in model_object["categorical_features"]:
    new_data[col] = new_data[col].astype(str)


for col in model_object["numeric_features"]:
    new_data[col] = pd.to_numeric(new_data[col], errors="coerce")


expected_features = model_object["numeric_features"] + model_object["categorical_features"]
new_data = new_data[expected_features]


pred_stage1 = model_object["stage1"].predict(new_data)
pred_stage2 = model_object["stage2"].predict(new_data)








extreme_scaling_factor = np.where(
    pred_stage1 >= 80, 100.0,
    np.where(pred_stage1 >= 40, 50.0, 10.0)
)
extreme_predictions = pred_stage1 + (extreme_scaling_factor * pred_stage2)




print("EXTREMELY Conditionally Scaled Predictions:")
for name, points in zip(player_names, extreme_predictions):
    print(f"{name}: {points:.2f}")
