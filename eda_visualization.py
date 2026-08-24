import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("dataset/loan_prediction_cleaned.csv")

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

# 1. Loan Status Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Loan_Status")
plt.title("Loan Approval Status")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("graphs/loan_status.png")
plt.show()

# 2. Applicant Income Distribution
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="ApplicantIncome", bins=30, kde=True)
plt.title("Applicant Income Distribution")
plt.xlabel("Applicant Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graphs/applicant_income.png")
plt.show()

# 3. Education vs Loan Status
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Education", hue="Loan_Status")
plt.title("Education vs Loan Approval")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("graphs/education_loan_status.png")
plt.show()

# 4. Credit History vs Loan Status
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Credit_History", hue="Loan_Status")
plt.title("Credit History vs Loan Approval")
plt.xlabel("Credit History")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("graphs/credit_history_loan_status.png")
plt.show()

# 5. Property Area vs Loan Status
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="Property_Area", hue="Loan_Status")
plt.title("Property Area vs Loan Approval")
plt.xlabel("Property Area")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("graphs/property_area_loan_status.png")
plt.show()

print("All graphs created successfully!")