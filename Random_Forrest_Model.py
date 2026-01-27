import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

# 1. Load and clean training data
df_train = pd.read_csv('Data/DraftClassData/all_drafted_RB.csv', encoding='latin1')
df_train = df_train.dropna(subset=['Tier'])

# 2. Define feature columns
features = ['Draft', 'Games', 'Carries', 'Rushing Yards', 'Yards per Carry', 'Rushing TD',
            'Rushing Yards per Game', 'Receptions', 'Receiving Yards', 'Yards per Reception',
            'Receiving TD', 'Receiving Yards per Game', 'Scrimmage Yards', 
            'Average Scrimmage Yards', 'Total TD']

# 3. Encode labels
X = df_train[features]
le = LabelEncoder()
y = le.fit_transform(df_train['Tier'])

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 5. Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 6. Define base model and grid search
base_rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}
grid = GridSearchCV(base_rf, param_grid, cv=5, scoring='f1_weighted')
grid.fit(X_resampled, y_resampled)

print("\nBest hyperparameters found:", grid.best_params_)

# 7. Calibrate the best model
calibrated_model = CalibratedClassifierCV(estimator=grid.best_estimator_, method='sigmoid', cv=5)
calibrated_model.fit(X_resampled, y_resampled)

# 8. Evaluate on the test set
y_pred = calibrated_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in le.classes_]))

# 9. Load prospect dataset and predict
df_prospects = pd.read_csv('Data/DraftClassData/2025_Drafted_RB.csv', encoding='latin1')
X_prospects = df_prospects[features]

predictions_encoded = calibrated_model.predict(X_prospects)
predicted_tiers = le.inverse_transform(predictions_encoded)

df_prospects['PredictedTier'] = predicted_tiers
print("\nProspect Predictions:")
print(df_prospects[['Name', 'PredictedTier']])

# 10. Feature Importances
importances = grid.best_estimator_.feature_importances_
importances_pct = importances * 100
indices = np.argsort(importances_pct)[::-1]
feature_names = [features[i] for i in indices]

print("\nFeature Importances (as %):")
for i, idx in enumerate(indices):
    print(f"{i + 1}. {features[idx]}: {importances_pct[idx]:.2f}%")

# 11. Plot Feature Importances
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(importances_pct)), importances_pct[indices], align='center')
plt.xticks(range(len(importances_pct)), feature_names, rotation=45, ha='right')
plt.title("Feature Importance (%)")
plt.ylabel("Importance (%)")
for bar, importance in zip(bars, importances_pct[indices]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{importance:.1f}%", 
             ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# 12. Tier distribution
print("\nTier counts (original):")
print(df_train['Tier'].value_counts())
plt.figure(figsize=(6,4))
df_train['Tier'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Original Tier Distribution')
plt.xlabel('Tier')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 13. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\nConfusion Matrix:")
print(cm_df)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

import tkinter as tk
from tkinter import ttk

def show_predictions_gui(df_prospects, probabilities, labels):
    window = tk.Tk()
    window.title("2025 Prospect Tier Predictions")

    # Create Treeview widget (like a table)
    tree = ttk.Treeview(window)
    
    # Define columns
    tree["columns"] = ("Tier", "P_T1", "P_T2", "P_T3", "P_T4")
    tree.heading("#0", text="Player")
    tree.heading("Tier", text="Predicted Tier")
    tree.heading("P_T1", text="T1 %")
    tree.heading("P_T2", text="T2 %")
    tree.heading("P_T3", text="T3 %")
    tree.heading("P_T4", text="T4 %")

    tree.column("#0", width=160)
    tree.column("Tier", width=100)
    for col in ["P_T1", "P_T2", "P_T3", "P_T4"]:
        tree.column(col, width=60, anchor="center")

    # Fill rows
    for i, row in df_prospects.iterrows():
        name = row["Name"]
        pred = row["PredictedTier"]
        prob = probabilities[i]
        tree.insert("", "end", text=name, values=(
            pred,
            f"{prob[0]*100:.1f}%",
            f"{prob[1]*100:.1f}%",
            f"{prob[2]*100:.1f}%",
            f"{prob[3]*100:.1f}%"
        ))

    tree.pack(expand=True, fill="both")
    window.mainloop()

# --- Add this block after your model has predicted for df_prospects ---

# Get predicted probabilities for the prospect dataset
prospect_probs = calibrated_model.predict_proba(X_prospects)

# Launch GUI window with predictions + probabilities
show_predictions_gui(df_prospects, prospect_probs, le.classes_)
