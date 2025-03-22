import pandas as pd
import numpy as np
import json
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
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

df = pd.read_csv("Data/data.csv")
label_column = "PPR Points"

df_labeled = df.dropna(subset=[label_column])   
df_unlabeled = df[df[label_column].isna()]      

X_raw_labeled = df_labeled.drop(columns=[label_column])
y_labeled = df_labeled[label_column]

X_labeled = pd.get_dummies(X_raw_labeled)

X_train, X_val, y_train, y_val = train_test_split(
    X_labeled, 
    y_labeled, 
    test_size=0.2, 
    random_state=42
)

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 5],
    'min_samples_split': [2, 5, 10]
}

tree = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=5,                         
    scoring='neg_mean_squared_error',
    n_jobs=-1                     
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best (CV) MSE score:", -grid_search.best_score_)

best_params = grid_search.best_params_
model = DecisionTreeRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

val_predictions = model.predict(X_val)

average_error = float(np.mean(val_predictions - y_val))

model_stats = {
    "Mean Squared Error (MSE)": float(mean_squared_error(y_val, val_predictions)),
    "Root Mean Squared Error (RMSE)": float(np.sqrt(mean_squared_error(y_val, val_predictions))),
    "Mean Absolute Error (MAE)": float(mean_absolute_error(y_val, val_predictions)),
    "Median Absolute Error (Median AE)": float(median_absolute_error(y_val, val_predictions)),
    "R^2 Score": float(r2_score(y_val, val_predictions)),
    "Explained Variance": float(explained_variance_score(y_val, val_predictions)),
    "Max Error": float(max_error(y_val, val_predictions)),
    "Mean Squared Log Error (MSLE)": float(mean_squared_log_error(y_val, val_predictions)),
    "Poisson Deviance": float(mean_poisson_deviance(y_val, val_predictions)),
    "Gamma Deviance": float(mean_gamma_deviance(y_val, val_predictions)),
    "Average Error (ME)": abs(average_error)
}

df_all_features = df.drop(columns=[label_column], errors='ignore')
X_all = pd.get_dummies(df_all_features)

X_all_aligned, X_train_aligned = X_all.align(X_labeled, join='left', axis=1)

X_all_aligned = X_all_aligned.fillna(0)

full_predictions = model.predict(X_all_aligned)

all_predictions = []

for i in range(len(df)):
    
    row_features = df_all_features.iloc[i].to_dict()
    
    row_dict = {
        **row_features,  
        "Predicted PPR Points": float(full_predictions[i])
    }
    
    actual_ppr = df[label_column].iloc[i]
    if pd.notna(actual_ppr):
        row_dict["Actual PPR Points"] = float(actual_ppr)
    else:
        row_dict["Actual PPR Points"] = None  

    all_predictions.append(row_dict)


all_predictions.append({"Model Statistics": model_stats})

os.makedirs("Predictions", exist_ok=True)
output_path = "Predictions/all_predictions.json"
with open(output_path, "w") as f:
    json.dump(all_predictions, f, indent=2)

print(f"Predictions saved to: {output_path}")