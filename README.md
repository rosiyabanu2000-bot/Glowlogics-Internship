# 📈 Sales Forecasting Using Machine Learning

## 📌 Project Overview

This project is a Sales Forecasting system developed using Python and Machine Learning.

The system analyzes historical sales transaction data and predicts future sales transaction volume using multiple machine learning algorithms.

## 🎯 Objective

The main objective of this project is to:

- Analyze historical sales data
- Perform data cleaning
- Perform Exploratory Data Analysis (EDA)
- Create useful visualizations
- Generate machine learning features
- Train multiple regression models
- Compare model performance
- Select the best-performing model
- Predict future sales transactions

## 📊 Dataset

The dataset contains sales transaction records with the following fields:

- `item_id` — Product/item identifier
- `date` — Transaction date
- `store_id` — Store identifier

Since the dataset does not contain a direct sales amount column, the number of transaction records per day is used as the sales target.

Therefore, this project performs **daily sales transaction forecasting**.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 🤖 Machine Learning Models

The following regression models were trained:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

## 📐 Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The model with the lowest RMSE was selected as the best model.

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Data Visualization
   ↓
Daily Sales Aggregation
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Multiple ML Models
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Prediction System