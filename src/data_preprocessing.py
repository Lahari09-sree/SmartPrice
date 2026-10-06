import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "sales_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed_sales_data.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original shape: {df.shape}")


# --------------------------------------------------
# BASIC CLEANING
# --------------------------------------------------

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

# Remove duplicate rows
duplicates = df.duplicated().sum()

print(f"Duplicate rows found: {duplicates}")

df = df.drop_duplicates()


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

# Date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Is weekend?
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)


# --------------------------------------------------
# PRICING FEATURES
# --------------------------------------------------

# Discounted selling price
df["Discounted_Price"] = (
    df["Price"] * (1 - df["Discount"] / 100)
)

# Price difference from competitor
df["Competitor_Price_Difference"] = (
    df["Price"] - df["Competitor Pricing"]
)

# Percentage difference from competitor
df["Competitor_Price_Gap_Percent"] = (
    (df["Price"] - df["Competitor Pricing"])
    / df["Competitor Pricing"]
) * 100


# --------------------------------------------------
# INVENTORY FEATURES
# --------------------------------------------------

# Inventory coverage based on current demand
df["Inventory_Demand_Ratio"] = (
    df["Inventory Level"]
    / df["Demand"].replace(0, np.nan)
)

# Inventory pressure
df["Inventory_Pressure"] = np.where(
    df["Inventory Level"] > df["Demand"],
    "High",
    "Normal"
)


# --------------------------------------------------
# ESTIMATED COST
# --------------------------------------------------

# Business assumption:
# Estimated product cost = 70% of listed price

COST_RATIO = 0.70

df["Estimated_Cost"] = df["Price"] * COST_RATIO

# Estimated profit using actual units sold
df["Actual_Revenue"] = (
    df["Units Sold"] * df["Discounted_Price"]
)

df["Estimated_Profit"] = (
    df["Actual_Revenue"]
    - (df["Units Sold"] * df["Estimated_Cost"])
)

# Profit margin
df["Profit_Margin_Percent"] = np.where(
    df["Actual_Revenue"] > 0,
    (df["Estimated_Profit"] / df["Actual_Revenue"]) * 100,
    0
)


# --------------------------------------------------
# SAVE PROCESSED DATA
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nProcessing completed!")

print(f"Final shape: {df.shape}")

print(f"Saved to: {OUTPUT_FILE}")

print("\nNew columns added:")

new_columns = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "IsWeekend",
    "Discounted_Price",
    "Competitor_Price_Difference",
    "Competitor_Price_Gap_Percent",
    "Inventory_Demand_Ratio",
    "Inventory_Pressure",
    "Estimated_Cost",
    "Actual_Revenue",
    "Estimated_Profit",
    "Profit_Margin_Percent"
]

print(new_columns)

print("\nSample processed data:")

print(
    df[
        [
            "Product ID",
            "Price",
            "Discount",
            "Discounted_Price",
            "Competitor Pricing",
            "Demand",
            "Inventory Level",
            "Estimated_Profit",
            "Profit_Margin_Percent"
        ]
    ].head()
)