# ==========================================
# PREDICTIVE ANALYTICS USING HISTORICAL DATA
# DATA INSPECTION, CLEANING & EDA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("airline_passengers.csv")

# Display First 5 Records
print("=" * 50)
print("FIRST 5 ROWS OF DATASET")
print("=" * 50)
print(df.head())

# Dataset Shape
print("\n" + "=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Data Types
print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

# Dataset Information
print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
df.info()

# Statistical Summary
print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

# Missing Values
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# Duplicate Records
print("\n" + "=" * 50)
print("DUPLICATE RECORDS")
print("=" * 50)
print("Duplicate Rows:", df.duplicated().sum())

# Convert Month Column to Datetime
df["Month"] = pd.to_datetime(df["Month"])
# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Extract Year and Month Number

df["Year"] = df["Month"].dt.year
df["Month_Number"] = df["Month"].dt.month

print("\n" + "=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

print(df.head())

# ==========================================
# PREPARE FEATURES
# ==========================================

X = df[["Year", "Month_Number"]]
y = df["Passengers"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# LINEAR REGRESSION MODEL
# ==========================================

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ==========================================
# MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

print("\nFirst 5 Predictions:")

for i in range(5):
    print(
        f"Actual: {y_test.iloc[i]} | Predicted: {predictions[i]:.2f}"
    )
    # ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Passengers")
plt.ylabel("Predicted Passengers")

plt.title("Actual vs Predicted Values")

plt.grid(True)

plt.savefig("actual_vs_predicted.png", dpi=300)

plt.close()

print("\nGraph Saved: actual_vs_predicted.png")
# ==========================================
# FUTURE FORECASTING
# ==========================================

future_data = pd.DataFrame({
    "Year": [1953],
    "Month_Number": [1]
})

future_prediction = model.predict(future_data)

print("\n" + "=" * 50)
print("FUTURE FORECAST")
print("=" * 50)

print(f"Predicted Passengers for January 1953: {future_prediction[0]:.2f}")
# ==========================================
# FUTURE FORECAST VISUALIZATION
# ==========================================

plt.figure(figsize=(8,5))

plt.bar(
    ["Jan 1953 Forecast"],
    [future_prediction[0]]
)

plt.title("Forecasted Airline Passengers")
plt.ylabel("Passengers")

plt.savefig("future_forecast.png", dpi=300)
plt.close()

print("Graph Saved: future_forecast.png")
