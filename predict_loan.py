import pandas as pd
import joblib

# Load saved model and feature names
model = joblib.load("best_loan_model.pkl")
feature_names = joblib.load("model_features.pkl")

print("====================================")
print("     LOAN APPROVAL PREDICTION")
print("====================================")

# Get applicant details
gender = input("Gender (Male/Female): ")
married = input("Married (Yes/No): ")
dependents = input("Dependents (0/1/2/3+): ")
education = input("Education (Graduate/Not Graduate): ")
self_employed = input("Self Employed (Yes/No): ")

applicant_income = float(input("Applicant Income: "))
coapplicant_income = float(input("Coapplicant Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Amount Term: "))
credit_history = float(input("Credit History (1/0): "))
property_area = input("Property Area (Urban/Semiurban/Rural): ")

# Create applicant dataframe
applicant = pd.DataFrame({
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Gender_Male": [1 if gender.lower() == "male" else 0],
    "Married_Yes": [1 if married.lower() == "yes" else 0],
    "Dependents_1": [1 if dependents == "1" else 0],
    "Dependents_2": [1 if dependents == "2" else 0],
    "Dependents_3+": [1 if dependents == "3+" else 0],
    "Education_Not Graduate": [
        1 if education.lower() == "not graduate" else 0
    ],
    "Self_Employed_Yes": [
        1 if self_employed.lower() == "yes" else 0
    ],
    "Property_Area_Semiurban": [
        1 if property_area.lower() == "semiurban" else 0
    ],
    "Property_Area_Urban": [
        1 if property_area.lower() == "urban" else 0
    ]
})

# Match training feature columns exactly
applicant = applicant.reindex(
    columns=feature_names,
    fill_value=0
)

# Prediction
prediction = model.predict(applicant)[0]

print("\n====================================")

if prediction == 1:
    print("      LOAN STATUS: APPROVED")
else:
    print("      LOAN STATUS: REJECTED")

print("====================================")