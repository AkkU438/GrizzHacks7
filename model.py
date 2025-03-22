import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
import joblib
import os
import numpy as np
import json

df = pd.read_csv("Data/Train.csv")
label_column = "PPR Points"
X_raw = df.drop(columns=[label_column])
y = df[label_column]
X = pd.get_dummies(X_raw)

model = DecisionTreeRegressor()
model.fit(X, y)

predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)
explained_var = explained_variance_score(y, predictions)

print("📊 MODEL EVALUATION STATS 📊")
print(f"MSE: {mse:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")
print(f"R²: {r2:.4f}  Explained Variance: {explained_var:.4f}")

importances = pd.Series(model.feature_importances_, index=X.columns)
least_useful = importances[importances == 0].index.tolist()
print(f"🗑️ Features with 0 importance: {least_useful}")

test_df = pd.read_csv("Data/Test.csv")

original_test_data = test_df.copy()

X_test_raw = pd.get_dummies(test_df)
X_test_aligned = X_test_raw.reindex(columns=X.columns, fill_value=0)
test_predictions = model.predict(X_test_aligned)


all_predictions = []

for i, row in original_test_data.iterrows():
    row_dict = row.to_dict()
    row_dict["Predicted PPR Points"] = float(test_predictions[i])
    all_predictions.append(row_dict)

os.makedirs("Predictions", exist_ok=True)
with open("Predictions/all_predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=2)

for i, row_data in enumerate(all_predictions):
    filename = f"Predictions/prediction_{i + 1}.json"
    with open(filename, "w") as f:
        json.dump(row_data, f, indent=2)

print("✅ Predictions exported to JSON files in the 'Predictions' folder.")
