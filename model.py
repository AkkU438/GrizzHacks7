import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os


train_path = "Data/Train.csv"
test_path = "Data/Test.csv"


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


label_column = "your_label_column"


X_train = train_df.drop(columns=[label_column])
y_train = train_df[label_column]

X_test = test_df.drop(columns=[label_column])
y_test = test_df[label_column]


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


print("Model depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


os.makedirs("ModelSaves", exist_ok=True)
joblib.dump(model, "ModelSaves/model.joblib")
