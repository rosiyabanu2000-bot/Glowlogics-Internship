
import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

print("Original Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing numerical values with median
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing categorical values with mode
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nAfter Cleaning:")
print("Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")