
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)

# Generate synthetic flight data
n = 5000

data = pd.DataFrame({
    "altitude": np.random.randint(500, 40000, n),
    "airspeed": np.random.randint(80, 600, n),
    "vertical_speed": np.random.randint(-5000, 5000, n),
    "pitch": np.random.uniform(-15, 20, n),
    "visibility": np.random.uniform(0.1, 15, n),
    "temperature": np.random.uniform(-60, 40, n)
})

# Synthetic labeling logic (for demonstration only)
unsafe = (
    (data["airspeed"] < 120) |
    (data["airspeed"] > 450) |
    (data["vertical_speed"] < -3500) |
    (data["vertical_speed"] > 3500) |
    (data["pitch"] < -10) |
    (data["pitch"] > 15) |
    (data["visibility"] < 1) |
    (data["altitude"] < 500)
)

data["safety_status"] = np.where(unsafe, 1, 0)

# 0 = Safe, 1 = Unsafe
X = data.drop("safety_status", axis=1)
y = data["safety_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save dataset
os.makedirs("dataset", exist_ok=True)
data.to_csv("dataset/aviation_data.csv", index=False)

# Save trained model
os.makedirs("model", exist_ok=True)

with open("model/aviation_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nDataset saved to dataset/aviation_data.csv")
print("Model saved to model/aviation_model.pkl")
print("Training completed successfully!")
