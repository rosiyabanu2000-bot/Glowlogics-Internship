import pandas as pd
import joblib

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load(
    "models/best_sales_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

print("=" * 60)
print("SALES FORECASTING - PREDICTION SYSTEM")
print("=" * 60)

# ============================================
# USER INPUT
# ============================================

year = int(input("Enter year: "))

month = int(
    input("Enter month (1-12): ")
)

day = int(
    input("Enter day (1-31): ")
)

day_of_week = int(
    input("Enter day of week (0=Monday, 6=Sunday): ")
)

week_of_year = int(
    input("Enter week of year: ")
)

sales_lag_1 = float(
    input("Enter previous day sales: ")
)

sales_lag_7 = float(
    input("Enter sales from 7 days ago: ")
)

sales_rolling_7 = float(
    input("Enter 7-day average sales: ")
)

sales_rolling_30 = float(
    input("Enter 30-day average sales: ")
)

# ============================================
# CREATE INPUT DATA
# ============================================

input_data = pd.DataFrame(
    [[
        year,
        month,
        day,
        day_of_week,
        week_of_year,
        sales_lag_1,
        sales_lag_7,
        sales_rolling_7,
        sales_rolling_30
    ]],
    columns=[
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
)

# Keep same feature order used during training
input_data = input_data[
    feature_columns
]

# ============================================
# PREDICTION
# ============================================

prediction = model.predict(
    input_data
)

# ============================================
# RESULT
# ============================================

print("\n" + "=" * 60)
print("📈 SALES FORECAST RESULT")
print("=" * 60)

print(
    "\nPredicted Sales Transactions:",
    round(prediction[0])
)

print("=" * 60)