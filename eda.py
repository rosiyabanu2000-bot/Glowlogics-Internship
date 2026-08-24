import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/loan_prediction_cleaned.csv")

print("===== DATASET INFORMATION =====")
df.info()

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== LOAN STATUS DISTRIBUTION =====")
print(df["Loan_Status"].value_counts())

print("\n===== CATEGORICAL COLUMNS =====")
print(df.select_dtypes(include="object").columns.tolist())

print("\n===== NUMERICAL COLUMNS =====")
print(df.select_dtypes(include="number").columns.tolist())