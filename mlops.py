import os
import csv
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_file = "logs/training_log.csv"

# Model versions
configs = [
    {"version": "v1", "max_iter": 100},
    {"version": "v2", "max_iter": 200},
    {"version": "v3", "max_iter": 300}
]

file_exists = os.path.isfile(log_file)

for config in configs:
    print("Training", config["version"])

    model = LogisticRegression(max_iter=config["max_iter"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    path = f"models/model_{config['version']}.pkl"
    joblib.dump(model, path)

    print("Saved:", path)
    print("Accuracy:", accuracy)

    # Log results
    with open(log_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["version", "max_iter", "accuracy"])

        if not file_exists:
            writer.writeheader()
            file_exists = True

        writer.writerow({
            "version": config["version"],
            "max_iter": config["max_iter"],
            "accuracy": accuracy
        })

print("All models trained successfully!")