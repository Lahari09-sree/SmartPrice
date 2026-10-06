import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed_sales_data.csv"
MODEL_FILE = BASE_DIR / "models" / "demand_model.pkl"


# ============================================================
# BUSINESS SETTINGS
# ============================================================

COST_RATIO = 0.70

MIN_PROFIT_MARGIN = 10.0

DISCOUNT_OPTIONS = [0, 5, 10, 15, 20, 25]


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

model = joblib.load(MODEL_FILE)


# ============================================================
# DEMAND PREDICTION FUNCTION
# ============================================================

def predict_demand(row, discount):

    input_data = pd.DataFrame(
        [{
            "Store ID": row["Store ID"],
            "Product ID": row["Product ID"],
            "Category": row["Category"],
            "Region": row["Region"],
            "Inventory Level": row["Inventory Level"],
            "Units Ordered": row["Units Ordered"],
            "Price": row["Price"],
            "Discount": discount,
            "Weather Condition": row["Weather Condition"],
            "Promotion": row["Promotion"],
            "Competitor Pricing": row["Competitor Pricing"],
            "Seasonality": row["Seasonality"],
            "Epidemic": row["Epidemic"],
            "Year": row["Date"].year,
            "Month": row["Date"].month,
            "DayOfWeek": row["Date"].dayofweek,
            "IsWeekend": int(row["Date"].dayofweek >= 5)
        }]
    )

    predicted = model.predict(input_data)[0]

    return max(0, predicted)


# ============================================================
# PRICING OPTIMIZER
# ============================================================

def optimize_price(row):

    results = []

    cost_per_unit = row["Price"] * COST_RATIO

    for discount in DISCOUNT_OPTIONS:

        # Discounted selling price
        selling_price = row["Price"] * (
            1 - discount / 100
        )

        # Predict demand
        predicted_demand = predict_demand(
            row,
            discount
        )

        # Cannot sell more than available inventory
        expected_units = min(
            predicted_demand,
            row["Inventory Level"]
        )

        # Revenue
        revenue = (
            expected_units
            * selling_price
        )

        # Cost
        total_cost = (
            expected_units
            * cost_per_unit
        )

        # Profit
        profit = revenue - total_cost

        # Margin
        if revenue > 0:
            margin = (
                profit / revenue
            ) * 100
        else:
            margin = 0

        # Feasibility
        feasible = (
            margin >= MIN_PROFIT_MARGIN
        )

        results.append(
            {
                "Discount": discount,
                "Selling Price": round(
                    selling_price,
                    2
                ),
                "Predicted Demand": round(
                    predicted_demand,
                    2
                ),
                "Expected Units": round(
                    expected_units,
                    2
                ),
                "Revenue": round(
                    revenue,
                    2
                ),
                "Profit": round(
                    profit,
                    2
                ),
                "Profit Margin": round(
                    margin,
                    2
                ),
                "Feasible": feasible
            }
        )

    results_df = pd.DataFrame(results)

    # Only consider feasible discounts
    feasible_results = results_df[
        results_df["Feasible"] == True
    ]

    if len(feasible_results) == 0:

        # No discount satisfies margin requirement
        recommendation = results_df.loc[
            results_df["Profit Margin"].idxmax()
        ]

        reason = (
            "No discount satisfied the minimum "
            "profit-margin requirement."
        )

    else:

        # Select maximum profit among feasible options
        recommendation = feasible_results.loc[
            feasible_results["Profit"].idxmax()
        ]

        reason = (
            "Selected the discount with the "
            "highest expected profit while "
            "maintaining the minimum profit margin."
        )

    return results_df, recommendation, reason


# ============================================================
# SELECT DEMO PRODUCT
# ============================================================

# Using the first record as demonstration
sample_row = df.iloc[0]


# ============================================================
# RUN OPTIMIZATION
# ============================================================

results, recommendation, reason = optimize_price(
    sample_row
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 80)
print("SMARTPRICE - PRICING OPTIMIZATION ENGINE")
print("=" * 80)

print("\nProduct Information")

print(
    f"Product ID       : {sample_row['Product ID']}"
)

print(
    f"Store ID         : {sample_row['Store ID']}"
)

print(
    f"Category         : {sample_row['Category']}"
)

print(
    f"Current Price    : ₹{sample_row['Price']:.2f}"
)

print(
    f"Inventory        : {sample_row['Inventory Level']}"
)

print(
    f"Competitor Price : ₹{sample_row['Competitor Pricing']:.2f}"
)

print(
    f"Current Discount : {sample_row['Discount']}%"
)


print("\n" + "=" * 80)
print("DISCOUNT SCENARIOS")
print("=" * 80)

print(
    results.to_string(index=False)
)


print("\n" + "=" * 80)
print("🏆 SMARTPRICE RECOMMENDATION")
print("=" * 80)

print(
    f"Recommended Discount : "
    f"{recommendation['Discount']}%"
)

print(
    f"Recommended Price    : "
    f"₹{recommendation['Selling Price']:.2f}"
)

print(
    f"Expected Demand      : "
    f"{recommendation['Predicted Demand']:.0f}"
)

print(
    f"Expected Units Sold  : "
    f"{recommendation['Expected Units']:.0f}"
)

print(
    f"Expected Revenue     : "
    f"₹{recommendation['Revenue']:,.2f}"
)

print(
    f"Expected Profit      : "
    f"₹{recommendation['Profit']:,.2f}"
)

print(
    f"Profit Margin        : "
    f"{recommendation['Profit Margin']:.2f}%"
)

print(
    f"\nReason: {reason}"
)