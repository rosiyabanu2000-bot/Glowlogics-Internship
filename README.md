# 🏦 Loan Approval Prediction System

## 📌 Project Overview

The Loan Approval Prediction System is a Machine Learning project
that predicts whether a loan application is likely to be approved
or rejected based on applicant information.

## 🎯 Objective

The objective of this project is to build a classification model
that can predict loan approval using applicant financial and
personal information.

## 📊 Dataset

The dataset contains 614 loan application records and 13 columns.

Important features include:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 🔄 Project Workflow

1. Dataset collection
2. Data loading
3. Data cleaning
4. Exploratory Data Analysis
5. Data visualization
6. Feature preparation
7. Model training
8. Model comparison
9. Best model selection
10. Loan prediction

## 🤖 Machine Learning Models

The following models were trained:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

## 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

## 🏆 Best Model

The best model was selected based on the highest F1 Score.

The actual model performance can be found in:

`model_results.csv`

## 🔮 Prediction System

The prediction system accepts a new applicant's information and
predicts:

- Loan Approved
- Loan Rejected

Run:

```bash
python predict_loan.py