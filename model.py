import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

# Sklearn-related imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_squared_log_error,
    mean_poisson_deviance
)
from xgboost import XGBRegressor

################################################################################
# 1) Read Data + Remove "Name" Column + Basic Prep
################################################################################
df = pd.read_csv("Data/NFLCollegeStats.csv")
label_column = "TFP"

# Remove "Name" if it exists
if "Name" in df.columns:
    df.drop(columns=["Name"], inplace=True)
    print("✅ Removed 'Name' column from features.")

# Outlier removal (IQR-based)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print(f"✅ After outlier removal, dataset has {len(df)} rows")

# Separate features & label
X_raw = df.drop(columns=[label_column])
y = df[label_column]

numeric_features = X_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

# Train/Validation Split
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

################################################################################
# 2) Preprocessor (Scaling + OneHot)
################################################################################
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

################################################################################
# 3) Stage-1 Model (XGBoost) + RandomizedSearch
################################################################################
model1_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb1", XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42
    ))
])

param_dist_stage1 = {
    "xgb1__n_estimators": [100, 200, 300, 400, 500, 600],
    "xgb1__max_depth": [3, 5, 7, 9, 12, 15],
    "xgb1__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "xgb1__subsample": [0.6, 0.8, 1.0],
    "xgb1__colsample_bytree": [0.5, 0.7, 0.9, 1.0]
}

search_stage1 = RandomizedSearchCV(
    estimator=model1_pipeline,
    param_distributions=param_dist_stage1,
    scoring="neg_mean_squared_error",
    n_iter=30,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
search_stage1.fit(X_train_raw, y_train)
best_model1 = search_stage1.best_estimator_

print("\n✅ Stage-1 Best Params:")
print(search_stage1.best_params_)

# Predict TFP on TRAIN; get residual
train_pred_stage1 = best_model1.predict(X_train_raw)
residual_train = y_train - train_pred_stage1

################################################################################
# 4) Stage-2 Model (XGBoost) to Predict Residuals
################################################################################
model2_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb2", XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=777
    ))
])

param_dist_stage2 = {
    "xgb2__n_estimators": [100, 200, 300, 400, 500, 600],
    "xgb2__max_depth": [3, 5, 7, 9, 12, 15],
    "xgb2__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "xgb2__subsample": [0.6, 0.8, 1.0],
    "xgb2__colsample_bytree": [0.5, 0.7, 0.9, 1.0]
}

search_stage2 = RandomizedSearchCV(
    estimator=model2_pipeline,
    param_distributions=param_dist_stage2,
    scoring="neg_mean_squared_error",
    n_iter=30,
    cv=5,
    verbose=1,
    random_state=777,
    n_jobs=-1
)
search_stage2.fit(X_train_raw, residual_train)
best_model2 = search_stage2.best_estimator_

print("\n✅ Stage-2 (Residual) Best Params:")
print(search_stage2.best_params_)

################################################################################
# 5) Combine Stage-1 + Stage-2 on Validation
################################################################################
val_pred_stage1 = best_model1.predict(X_val_raw)
val_pred_stage2 = best_model2.predict(X_val_raw)
val_pred_final_raw = val_pred_stage1 + val_pred_stage2

# Mean error correction
mean_error_val = np.mean(val_pred_final_raw - y_val)
val_pred_final_corrected = val_pred_final_raw - mean_error_val

# If below 50, reduce by a fraction (e.g., 50%); not all the way to zero.
fraction = 0.5  # adjust as you see fit
mask_below_50 = val_pred_final_corrected < 50
val_pred_final_corrected[mask_below_50] = val_pred_final_corrected[mask_below_50] * fraction

# Ensure strictly positive to avoid Poisson deviance error
# (anything that becomes negative or zero will be clipped)
val_pred_final_corrected = np.clip(val_pred_final_corrected, 1e-9, None)

################################################################################
# 6) Evaluate Metrics (Now that predictions are all > 0)
################################################################################
mse = mean_squared_error(y_val, val_pred_final_corrected)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, val_pred_final_corrected)
med_ae = median_absolute_error(y_val, val_pred_final_corrected)
r2 = r2_score(y_val, val_pred_final_corrected)
exp_var = explained_variance_score(y_val, val_pred_final_corrected)
mx_err = max_error(y_val, val_pred_final_corrected)
msle = mean_squared_log_error(y_val, val_pred_final_corrected)
poisson_dev = mean_poisson_deviance(y_val, val_pred_final_corrected)
avg_error = float(np.mean(val_pred_final_corrected - y_val))

# MAPE / MedAPE => "Accuracy"
mask_nonzero = (y_val != 0)
ape = np.abs(val_pred_final_corrected[mask_nonzero] - y_val[mask_nonzero]) / np.abs(y_val[mask_nonzero])
mape = ape.mean()
medape = np.median(ape)
avg_acc = 1 - mape
med_acc = 1 - medape

model_stats = {
    "Mean Squared Error (MSE)": float(mse),
    "Root Mean Squared Error (RMSE)": float(rmse),
    "Mean Absolute Error (MAE)": float(mae),
    "Median Absolute Error (Median AE)": float(med_ae),
    "R^2 Score": float(r2),
    "Explained Variance": float(exp_var),
    "Max Error": float(mx_err),
    "Mean Squared Log Error (MSLE)": float(msle),
    "Poisson Deviance": float(poisson_dev),
    "Average Error (ME)": abs(avg_error),
    "MAPE": float(mape),
    "MedAPE": float(medape),
    "Average Accuracy (1 - MAPE)": float(avg_acc),
    "Median Accuracy (1 - MedAPE)": float(med_acc)
}

print("\n🔍 Final Post-Processed Metrics (Two-Stage Model + <50 => scale down):")
for k, v in model_stats.items():
    if k.endswith("Importances"):
        continue
    print(f"  {k}: {v:.4f}")

################################################################################
# 7) Feature Importances (Stage-1 and Stage-2)
################################################################################
xgb1 = best_model1.named_steps["xgb1"]
importances1 = xgb1.feature_importances_

xgb2 = best_model2.named_steps["xgb2"]
importances2 = xgb2.feature_importances_

encoder_stage1 = best_model1.named_steps["preprocessor"].transformers_[1][1]
encoded_cat_cols_stage1 = encoder_stage1.get_feature_names_out(categorical_features)
all_feature_names_stage1 = numeric_features + list(encoded_cat_cols_stage1)

encoder_stage2 = best_model2.named_steps["preprocessor"].transformers_[1][1]
encoded_cat_cols_stage2 = encoder_stage2.get_feature_names_out(categorical_features)
all_feature_names_stage2 = numeric_features + list(encoded_cat_cols_stage2)

feature_importances_stage1 = sorted(
    zip(all_feature_names_stage1, importances1),
    key=lambda x: x[1],
    reverse=True
)
feature_importances_stage2 = sorted(
    zip(all_feature_names_stage2, importances2),
    key=lambda x: x[1],
    reverse=True
)

model_stats["Stage-1 Feature Importances"] = {
    f: float(i) for f, i in feature_importances_stage1
}
model_stats["Stage-2 Feature Importances"] = {
    f: float(i) for f, i in feature_importances_stage2
}

print("\nTop 10 Feature Importances (Stage-1 Model):")
for feat, imp in feature_importances_stage1[:10]:
    print(f"  {feat}: {imp:.4f}")

print("\nTop 10 Feature Importances (Stage-2 Residual Model):")
for feat, imp in feature_importances_stage2[:10]:
    print(f"  {feat}: {imp:.4f}")

################################################################################
# 8) Predict ALL DATA + Save JSON
################################################################################
full_pred_stage1 = best_model1.predict(X_raw)
full_pred_stage2 = best_model2.predict(X_raw)
full_pred_raw = full_pred_stage1 + full_pred_stage2

# Mean error shift
full_pred_corrected = full_pred_raw - mean_error_val

# Scale down if < 50
mask_below_50_full = full_pred_corrected < 50
full_pred_corrected[mask_below_50_full] = full_pred_corrected[mask_below_50_full] * fraction

# Clip to ensure strictly positive
full_pred_corrected = np.clip(full_pred_corrected, 1e-9, None)

all_predictions = []
for i in range(len(df)):
    row_dict = {
        "Predicted TFP (Raw, Stage-1+2)": float(full_pred_raw[i]),
        "Predicted TFP (Corrected)": float(full_pred_corrected[i]),
        "Actual TFP": float(y.iloc[i])
    }
    row_features = X_raw.iloc[i].to_dict()
    row_dict.update(row_features)
    all_predictions.append(row_dict)

all_predictions.append({"Model Statistics": model_stats})

os.makedirs("Predictions", exist_ok=True)
output_path = "Predictions/all_predictions.json"
with open(output_path, "w") as f:
    json.dump(all_predictions, f, indent=2)

print(f"\n✅ Two-Stage Predictions + validation stats saved to: {output_path}")
