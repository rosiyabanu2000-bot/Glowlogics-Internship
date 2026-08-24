import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# SETUP
# ============================================================

os.makedirs("graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 70)
print("SALES FORECASTING PROJECT")
print("=" * 70)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

df = pd.read_csv("dataset/sales_data.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# STEP 2 - DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - DATA CLEANING")
print("=" * 70)

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Convert date
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

# Remove invalid dates
df = df.dropna(
    subset=["date"]
)

# Remove duplicates
df = df.drop_duplicates()

# Sort by date
df = df.sort_values("date")

df = df.reset_index(drop=True)

print("\nCleaned Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# Save cleaned data
df.to_csv(
    "dataset/cleaned_sales_data.csv",
    index=False
)

print("\n✅ Cleaned dataset saved.")


# ============================================================
# STEP 3 - CREATE SALES TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - CREATE DAILY SALES TARGET")
print("=" * 70)

# Count records for each date
daily_sales = (
    df.groupby("date")
    .size()
    .reset_index(name="sales")
)

print("\nDaily Sales Data:")
print(daily_sales.head())

print("\nDaily Sales Shape:")
print(daily_sales.shape)

print("\nTotal Sales Transactions:")
print(daily_sales["sales"].sum())

print("\nAverage Daily Sales:")
print(daily_sales["sales"].mean())


# ============================================================
# STEP 4 - EDA
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - EDA")
print("=" * 70)

print("\nSales Statistics:")
print(daily_sales["sales"].describe())


# ============================================================
# STEP 5 - GRAPH 1: SALES OVER TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"]
)

plt.title("Daily Sales Transactions Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "graphs/1_sales_over_time.png"
)

plt.close()

print("\n✅ Graph 1 created.")


# ============================================================
# STEP 6 - GRAPH 2: SALES DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    daily_sales["sales"],
    bins=30
)

plt.title("Daily Sales Distribution")
plt.xlabel("Daily Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "graphs/2_sales_distribution.png"
)

plt.close()

print("✅ Graph 2 created.")


# ============================================================
# STEP 7 - MONTHLY SALES
# ============================================================

daily_sales["month"] = (
    daily_sales["date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = (
    daily_sales
    .groupby("month")["sales"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["month"],
    monthly_sales["sales"],
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "graphs/3_monthly_sales.png"
)

plt.close()

print("✅ Graph 3 created.")


# ============================================================
# STEP 8 - FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - ML FEATURE ENGINEERING")
print("=" * 70)

# Date features

daily_sales["year"] = (
    daily_sales["date"].dt.year
)

daily_sales["month_number"] = (
    daily_sales["date"].dt.month
)

daily_sales["day"] = (
    daily_sales["date"].dt.day
)

daily_sales["day_of_week"] = (
    daily_sales["date"].dt.dayofweek
)

daily_sales["week_of_year"] = (
    daily_sales["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

# Lag features

daily_sales["sales_lag_1"] = (
    daily_sales["sales"].shift(1)
)

daily_sales["sales_lag_7"] = (
    daily_sales["sales"].shift(7)
)

# Rolling averages

daily_sales["sales_rolling_7"] = (
    daily_sales["sales"]
    .rolling(7)
    .mean()
)

daily_sales["sales_rolling_30"] = (
    daily_sales["sales"]
    .rolling(30)
    .mean()
)

# Remove NaN values

daily_sales = daily_sales.dropna()

daily_sales = daily_sales.reset_index(
    drop=True
)


# Save ML dataset

daily_sales.to_csv(
    "dataset/ml_sales_data.csv",
    index=False
)

print("\nML dataset saved.")


# ============================================================
# STEP 9 - PREPARE FEATURES
# ============================================================

feature_columns = [
    "year",
    "month_number",
    "day",
    "day_of_week",
    "week_of_year",
    "sales_lag_1",
    "sales_lag_7",
    "sales_rolling_7",
    "sales_rolling_30"
]

X = daily_sales[
    feature_columns
]

y = daily_sales[
    "sales"
]

print("\nFeatures:")
print(feature_columns)

print("\nX Shape:")
print(X.shape)

print("\nY Shape:")
print(y.shape)


# ============================================================
# STEP 10 - TIME BASED TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 - TRAIN TEST SPLIT")
print("=" * 70)

# Time series should be split chronologically

split_index = int(
    len(daily_sales) * 0.80
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]

print("\nTraining rows:")
print(len(X_train))

print("Testing rows:")
print(len(X_test))


# ============================================================
# STEP 11 - TRAIN MULTIPLE MODELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 - TRAIN MULTIPLE MODELS")
print("=" * 70)

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
}


results = {}

trained_models = {}


for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[name] = model

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))


# ============================================================
# STEP 12 - MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 - MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    results
).T

print("\n")
print(results_df)

results_df.to_csv(
    "model_comparison.csv"
)


# ============================================================
# STEP 13 - SELECT BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 - BEST MODEL")
print("=" * 70)

# Lowest RMSE = best model

best_model_name = (
    results_df["RMSE"]
    .idxmin()
)

best_model = trained_models[
    best_model_name
]

print("\n🏆 Best Model:")
print(best_model_name)

print(
    "\nMAE:",
    round(
        results_df.loc[
            best_model_name,
            "MAE"
        ],
        2
    )
)

print(
    "RMSE:",
    round(
        results_df.loc[
            best_model_name,
            "RMSE"
        ],
        2
    )
)

print(
    "R²:",
    round(
        results_df.loc[
            best_model_name,
            "R2"
        ],
        4
    )
)


# ============================================================
# STEP 14 - SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/best_sales_model.pkl"
)

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("\n✅ Best model saved.")
print("models/best_sales_model.pkl")
print("models/feature_columns.pkl")


# ============================================================
# STEP 15 - MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df.index,
    results_df["RMSE"]
)

plt.title(
    "Model Comparison - RMSE"
)

plt.xlabel("Model")

plt.ylabel("RMSE")

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    "graphs/4_model_comparison.png"
)

plt.close()

print("✅ Model comparison graph created.")


# ============================================================
# STEP 16 - ACTUAL VS PREDICTED
# ============================================================

best_predictions = best_model.predict(
    X_test
)

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    best_predictions
)

plt.title(
    "Actual Sales vs Predicted Sales"
)

plt.xlabel(
    "Actual Sales"
)

plt.ylabel(
    "Predicted Sales"
)

plt.tight_layout()

plt.savefig(
    "graphs/5_actual_vs_predicted.png"
)

plt.close()

print("✅ Actual vs predicted graph created.")


# ============================================================
# STEP 17 - FINAL
# ============================================================

print("\n" + "=" * 70)
print("🎉 SALES FORECASTING PROJECT COMPLETED")
print("=" * 70)

print("\nGenerated Files:")

print("✅ dataset/cleaned_sales_data.csv")
print("✅ dataset/ml_sales_data.csv")
print("✅ model_comparison.csv")
print("✅ models/best_sales_model.pkl")
print("✅ models/feature_columns.pkl")

print("\nGraphs:")
print("✅ graphs/1_sales_over_time.png")
print("✅ graphs/2_sales_distribution.png")
print("✅ graphs/3_monthly_sales.png")
print("✅ graphs/4_model_comparison.png")
print("✅ graphs/5_actual_vs_predicted.png")

print("\n🎯 All ML steps completed successfully!")