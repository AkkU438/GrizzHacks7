import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_squared_log_error,
    mean_poisson_deviance,
    mean_gamma_deviance
)
import numpy as np
import json
import os

# === 1. Load combined dataset ===
df = pd.read_csv("Data/data.csv")
label_column = "PPR Points"

# Split features and label
X_raw = df.drop(columns=[label_column])
y = df[label_column]

# One-hot encode all categorical features
X = pd.get_dummies(X_raw)

# === 2. Split into train and validation sets ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Train model (with some constraints to reduce overfitting) ===
model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)

# === 4. Predict on validation set for evaluation ===
val_predictions = model.predict(X_val)

# === 5. Compute model statistics (on validation set) ===
# Average error (ME) = mean(prediction - actual)
average_error = float(np.mean(val_predictions - y_val))

model_stats = {
    "Mean Squared Error (MSE)": float(mean_squared_error(y_val, val_predictions)),
    "Root Mean Squared Error (RMSE)": float(
        np.sqrt(mean_squared_error(y_val, val_predictions))
    ),
    "Mean Absolute Error (MAE)": float(mean_absolute_error(y_val, val_predictions)),
    "Median Absolute Error (Median AE)": float(
        median_absolute_error(y_val, val_predictions)
    ),
    "R^2 Score": float(r2_score(y_val, val_predictions)),
    "Explained Variance": float(explained_variance_score(y_val, val_predictions)),
    "Max Error": float(max_error(y_val, val_predictions)),
    "Mean Squared Log Error (MSLE)": float(mean_squared_log_error(y_val, val_predictions)),
    "Poisson Deviance": float(mean_poisson_deviance(y_val, val_predictions)),
    "Gamma Deviance": float(mean_gamma_deviance(y_val, val_predictions)),
    "Average Error (ME)": average_error
}

# === 6. Make predictions on the *full* dataset ===
full_predictions = model.predict(X)

# === 7. Build JSON output ===
all_predictions = []
for i in range(len(df)):
    row_dict = {
        "Predicted PPR Points": float(full_predictions[i]),
        "Actual PPR Points": float(y.iloc[i])
    }
    # Add original non-encoded features
    row_features = X_raw.iloc[i].to_dict()
    row_dict.update(row_features)

    all_predictions.append(row_dict)

# Append validation stats at the end
all_predictions.append({"Model Statistics": model_stats})

# === 8. Save to JSON ===
os.makedirs("Predictions", exist_ok=True)
output_path = "Predictions/all_predictions.json"
with open(output_path, "w") as f:
    json.dump(all_predictions, f, indent=2)

print(f"✅ Predictions + validation stats (including average error) saved to: {output_path}")
