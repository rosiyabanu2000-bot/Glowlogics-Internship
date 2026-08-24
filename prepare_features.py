import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("dataset/loan_prediction_cleaned.csv")

# Convert target variable
df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})

# Convert categorical features into numerical features
df = pd.get_dummies(
    df,
    columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area"
    ],
    drop_first=True
)

# Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())

# Save processed datasets
X_train.to_csv("dataset/X_train.csv", index=False)
X_test.to_csv("dataset/X_test.csv", index=False)
y_train.to_csv("dataset/y_train.csv", index=False)
y_test.to_csv("dataset/y_test.csv", index=False)

print("\nML features prepared successfully!")