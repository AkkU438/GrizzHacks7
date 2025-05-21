import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Load the training dataset (current NFL players)
df_train = pd.read_csv('Data/NFLCollegeStatsCSV.csv', encoding='latin1')

# 2. Drop rows where 'Tier' is missing (only train on labeled data)
df_train = df_train.dropna(subset=['Tier'])

# 3. Define your feature columns (update to match your data)
features = ['Games', 'Carries', 'Rushing Yards', 'Yards per Carry', 'Rushing TD', 'Rushing Yards per Game', 'Receptions', 'Receiving Yards', 'Yards per Reception', 'Receiving TD', 'Receiving Yards per Game', 'Total Yards', 'Total Yards per Play', 'Total TD', 'Years']

# 4. Prepare feature matrix and label vector
X = df_train[features]
le = LabelEncoder()
y = le.fit_transform(df_train['Tier'])  # e.g., RB1 → 0, RB2 → 1, etc.

# 5. Train/test split for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 6. Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in le.classes_]))

# 8. Load the prospect dataset (players you want to predict for)
df_prospects = pd.read_csv('Data/nfl_draft_prospects_fixed.csv', encoding='latin1')

# 9. Make sure the same feature columns exist in this dataset
X_prospects = df_prospects[features]

# 10. Predict tiers (encoded), then decode to labels
predictions_encoded = model.predict(X_prospects)
predicted_tiers = le.inverse_transform(predictions_encoded)

# 11. Add predictions to the prospect DataFrame
df_prospects['PredictedTier'] = predicted_tiers

# 12. View or export predictions
print(df_prospects[['Name', 'PredictedTier']])
# Optionally save to file
df_prospects.to_csv('Data/predicted_prospect_tiers.csv', index=False)
