import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load training data
X_train = pd.read_csv("dataset/X_train.csv")
y_train = pd.read_csv("dataset/y_train.csv").squeeze()

# Load model results
results = pd.read_csv("model_results.csv")

# Select best model based on F1 Score
best_model_name = results.loc[
    results["F1 Score"].idxmax(), "Model"
]

print("Best Model:", best_model_name)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Get best model
best_model = models[best_model_name]

# Train best model using training data
best_model.fit(X_train, y_train)

# Save model
joblib.dump(best_model, "best_loan_model.pkl")

# Save feature names
joblib.dump(X_train.columns.tolist(), "model_features.pkl")

print("Best model saved successfully!")
print("Model file: best_loan_model.pkl")