import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load training and testing data
X_train = pd.read_csv("dataset/X_train.csv")
X_test = pd.read_csv("dataset/X_test.csv")

y_train = pd.read_csv("dataset/y_train.csv").squeeze()
y_test = pd.read_csv("dataset/y_test.csv").squeeze()

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

results = []

# Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Display results
results_df = pd.DataFrame(results)

print("\n===== MODEL PERFORMANCE =====")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("model_results.csv", index=False)

print("\nModel training completed!")
print("Results saved to model_results.csv")