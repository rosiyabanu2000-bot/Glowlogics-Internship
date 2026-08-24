import pandas as pd

# Load dataset
df = pd.read_csv("dataset/loan_prediction.csv")

print("Original Shape:")
print(df.shape)

# Remove unnecessary Loan_ID column
df = df.drop("Loan_ID", axis=1)

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill categorical missing values
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Fill numerical missing values
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].mode()[0]
)
df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].mode()[0]
)

# Remove duplicate rows
df = df.drop_duplicates()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Shape:")
print(df.shape)

# Save cleaned dataset
df.to_csv("dataset/loan_prediction_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")