import pandas as pd
import numpy as np
import os
import warnings
import joblib
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

# Predict on TRAIN; compute residuals
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
# 5) Combine Stage-1 + Stage-2 on Validation (Raw Predictions)
################################################################################
val_pred_stage1 = best_model1.predict(X_val_raw)
val_pred_stage2 = best_model2.predict(X_val_raw)
# Raw combined prediction (without any bias correction)
final_val_pred = val_pred_stage1 + val_pred_stage2
final_val_pred = np.clip(final_val_pred, 1e-9, None)  # ensure positivity

################################################################################
# 6) Evaluate Metrics (Using Raw Predictions)
################################################################################
mse = mean_squared_error(y_val, final_val_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, final_val_pred)
med_ae = median_absolute_error(y_val, final_val_pred)
r2 = r2_score(y_val, final_val_pred)
exp_var = explained_variance_score(y_val, final_val_pred)
mx_err = max_error(y_val, final_val_pred)
msle = mean_squared_log_error(y_val, final_val_pred)
poisson_dev = mean_poisson_deviance(y_val, final_val_pred)
avg_error = np.mean(final_val_pred - y_val)

mask_nonzero = (y_val != 0)
ape = np.abs(final_val_pred[mask_nonzero] - y_val[mask_nonzero]) / np.abs(y_val[mask_nonzero])
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

print("\n🔍 Final Metrics (using raw stage-1 + stage-2 predictions):")
for k, v in model_stats.items():
    print(f"  {k}: {v:.4f}")

################################################################################
# 7) Save the Model to Joblib
################################################################################
# Save the trained stage-1 and stage-2 models as a dictionary
model_object = {
    "stage1": best_model1,
    "stage2": best_model2,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}
os.makedirs("ModelSaves", exist_ok=True)
model_save_path = "ModelSaves/model.joblib"
joblib.dump(model_object, model_save_path)
print(f"\n✅ Model saved to: {model_save_path}")
