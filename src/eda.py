import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed_sales_data.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("SMARTPRICE - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("\nCategories:")
print(df["Category"].value_counts())

print("\nRegions:")
print(df["Region"].value_counts())

print("\nStores:")
print(df["Store ID"].nunique())

print("\nProducts:")
print(df["Product ID"].nunique())

# --------------------------------------------------
# OVERALL BUSINESS METRICS
# --------------------------------------------------

total_units = df["Units Sold"].sum()
total_demand = df["Demand"].sum()
total_revenue = df["Actual_Revenue"].sum()
total_profit = df["Estimated_Profit"].sum()

avg_discount = df["Discount"].mean()
avg_price = df["Price"].mean()
avg_margin = df["Profit_Margin_Percent"].mean()

print("\n" + "=" * 60)
print("OVERALL BUSINESS METRICS")
print("=" * 60)

print(f"Total Units Sold       : {total_units:,}")
print(f"Total Demand           : {total_demand:,}")
print(f"Total Revenue          : ₹{total_revenue:,.2f}")
print(f"Estimated Profit       : ₹{total_profit:,.2f}")
print(f"Average Price          : ₹{avg_price:,.2f}")
print(f"Average Discount       : {avg_discount:.2f}%")
print(f"Average Profit Margin  : {avg_margin:.2f}%")

# --------------------------------------------------
# CATEGORY ANALYSIS
# --------------------------------------------------

category_analysis = (
    df.groupby("Category")
    .agg(
        Units_Sold=("Units Sold", "sum"),
        Demand=("Demand", "sum"),
        Revenue=("Actual_Revenue", "sum"),
        Profit=("Estimated_Profit", "sum"),
        Avg_Discount=("Discount", "mean"),
        Avg_Margin=("Profit_Margin_Percent", "mean")
    )
    .sort_values("Profit", ascending=False)
)

print("\n" + "=" * 60)
print("CATEGORY PERFORMANCE")
print("=" * 60)

print(category_analysis.round(2))

# --------------------------------------------------
# DISCOUNT ANALYSIS
# --------------------------------------------------

discount_analysis = (
    df.groupby("Discount")
    .agg(
        Avg_Demand=("Demand", "mean"),
        Avg_Units_Sold=("Units Sold", "mean"),
        Revenue=("Actual_Revenue", "sum"),
        Profit=("Estimated_Profit", "sum"),
        Avg_Margin=("Profit_Margin_Percent", "mean")
    )
    .sort_index()
)

print("\n" + "=" * 60)
print("DISCOUNT ANALYSIS")
print("=" * 60)

print(discount_analysis.round(2))

# --------------------------------------------------
# PROMOTION ANALYSIS
# --------------------------------------------------

promotion_analysis = (
    df.groupby("Promotion")
    .agg(
        Avg_Demand=("Demand", "mean"),
        Avg_Units_Sold=("Units Sold", "mean"),
        Revenue=("Actual_Revenue", "sum"),
        Profit=("Estimated_Profit", "sum")
    )
)

print("\n" + "=" * 60)
print("PROMOTION ANALYSIS")
print("=" * 60)

print(promotion_analysis.round(2))

# --------------------------------------------------
# SEASONAL ANALYSIS
# --------------------------------------------------

season_analysis = (
    df.groupby("Seasonality")
    .agg(
        Avg_Demand=("Demand", "mean"),
        Units_Sold=("Units Sold", "sum"),
        Revenue=("Actual_Revenue", "sum"),
        Profit=("Estimated_Profit", "sum")
    )
    .sort_values("Profit", ascending=False)
)

print("\n" + "=" * 60)
print("SEASONAL PERFORMANCE")
print("=" * 60)

print(season_analysis.round(2))

# --------------------------------------------------
# COMPETITOR ANALYSIS
# --------------------------------------------------

df["Our_Price_vs_Competitor"] = np.where(
    df["Price"] < df["Competitor Pricing"],
    "Cheaper",
    np.where(
        df["Price"] > df["Competitor Pricing"],
        "More Expensive",
        "Same Price"
    )
)

competitor_analysis = (
    df.groupby("Our_Price_vs_Competitor")
    .agg(
        Avg_Demand=("Demand", "mean"),
        Avg_Units_Sold=("Units Sold", "mean"),
        Revenue=("Actual_Revenue", "sum"),
        Profit=("Estimated_Profit", "sum")
    )
)

print("\n" + "=" * 60)
print("COMPETITOR PRICE ANALYSIS")
print("=" * 60)

print(competitor_analysis.round(2))

# --------------------------------------------------
# INVENTORY ANALYSIS
# --------------------------------------------------

inventory_analysis = (
    df.groupby("Inventory_Pressure")
    .agg(
        Avg_Inventory=("Inventory Level", "mean"),
        Avg_Demand=("Demand", "mean"),
        Avg_Units_Sold=("Units Sold", "mean"),
        Profit=("Estimated_Profit", "sum")
    )
)

print("\n" + "=" * 60)
print("INVENTORY ANALYSIS")
print("=" * 60)

print(inventory_analysis.round(2))

# --------------------------------------------------
# HIGH INVENTORY / LOW DEMAND PRODUCTS
# --------------------------------------------------

product_summary = (
    df.groupby("Product ID")
    .agg(
        Avg_Inventory=("Inventory Level", "mean"),
        Avg_Demand=("Demand", "mean"),
        Avg_Discount=("Discount", "mean"),
        Avg_Price=("Price", "mean"),
        Total_Units_Sold=("Units Sold", "sum"),
        Total_Profit=("Estimated_Profit", "sum")
    )
)

# Overstock score
product_summary["Inventory_Demand_Ratio"] = (
    product_summary["Avg_Inventory"]
    / product_summary["Avg_Demand"].replace(0, np.nan)
)

overstocked = product_summary.sort_values(
    "Inventory_Demand_Ratio",
    ascending=False
).head(10)

print("\n" + "=" * 60)
print("TOP 10 POTENTIALLY OVERSTOCKED PRODUCTS")
print("=" * 60)

print(overstocked.round(2))

# --------------------------------------------------
# HIGH DEMAND / LOW INVENTORY
# --------------------------------------------------

high_demand_low_inventory = product_summary.sort_values(
    ["Avg_Demand", "Avg_Inventory"],
    ascending=[False, True]
).head(10)

print("\n" + "=" * 60)
print("HIGH DEMAND / LOW INVENTORY PRODUCTS")
print("=" * 60)

print(high_demand_low_inventory.round(2))

# --------------------------------------------------
# CORRELATIONS
# --------------------------------------------------

numeric_columns = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
    "Demand",
    "Estimated_Profit"
]

correlation = df[numeric_columns].corr()["Demand"].sort_values(
    ascending=False
)

print("\n" + "=" * 60)
print("CORRELATION WITH DEMAND")
print("=" * 60)

print(correlation.round(3))

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)