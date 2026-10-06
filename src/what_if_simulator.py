import os
import joblib
import pandas as pd

# ============================================================
# SMARTPRICE - WHAT-IF PRICING SIMULATOR
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_sales_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "demand_model.pkl"
)

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# ============================================================
# SETTINGS
# ============================================================

COST_RATIO = 0.70
MIN_PROFIT_MARGIN = 10

DISCOUNT_OPTIONS = [0, 5, 10, 15, 20, 25]


# ============================================================
# DEMAND PREDICTION
# ============================================================

def predict_demand(row, discount):

    features = pd.DataFrame([{
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
        "Year": row["Year"],
        "Month": row["Month"],
        "DayOfWeek": row["DayOfWeek"],
        "IsWeekend": row["IsWeekend"]
    }])

    prediction = model.predict(features)[0]

    return max(0, float(prediction))


# ============================================================
# SIMULATE DISCOUNTS
# ============================================================

def simulate(row):

    results = []

    price = float(row["Price"])
    inventory = float(row["Inventory Level"])

    for discount in DISCOUNT_OPTIONS:

        selling_price = price * (1 - discount / 100)

        demand = predict_demand(
            row,
            discount
        )

        expected_units = min(
            demand,
            inventory
        )

        revenue = (
            expected_units *
            selling_price
        )

        cost_per_unit = (
            price *
            COST_RATIO
        )

        total_cost = (
            expected_units *
            cost_per_unit
        )

        profit = (
            revenue -
            total_cost
        )

        margin = (
            profit / revenue * 100
            if revenue > 0
            else 0
        )

        inventory_clearance = (
            expected_units /
            inventory *
            100
            if inventory > 0
            else 0
        )

        feasible = (
            margin >= MIN_PROFIT_MARGIN
        )

        results.append({

            "Discount": discount,

            "Selling Price": selling_price,

            "Predicted Demand": demand,

            "Expected Units": expected_units,

            "Revenue": revenue,

            "Profit": profit,

            "Profit Margin": margin,

            "Inventory Clearance": inventory_clearance,

            "Feasible": feasible
        })

    return pd.DataFrame(results)


# ============================================================
# FIND BEST PROFITABLE SCENARIO
# ============================================================

def find_recommendation(results):

    feasible = results[
        results["Feasible"] == True
    ]

    if len(feasible) > 0:

        best_index = feasible[
            "Profit"
        ].idxmax()

    else:

        best_index = results[
            "Profit Margin"
        ].idxmax()

    return results.loc[best_index]


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 90)
    print("SMARTPRICE - WHAT-IF PRICING SIMULATOR")
    print("=" * 90)

    # --------------------------------------------------------
    # Select product
    # --------------------------------------------------------

    sample_row = df.iloc[0]

    print("\nPRODUCT")
    print("-" * 90)

    print(
        f"Product ID       : "
        f"{sample_row['Product ID']}"
    )

    print(
        f"Category         : "
        f"{sample_row['Category']}"
    )

    print(
        f"Current Price    : "
        f"₹{sample_row['Price']:.2f}"
    )

    print(
        f"Inventory        : "
        f"{sample_row['Inventory Level']:.0f}"
    )

    print(
        f"Competitor Price : "
        f"₹{sample_row['Competitor Pricing']:.2f}"
    )

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    results = simulate(sample_row)

    print("\n" + "=" * 90)
    print("WHAT-IF DISCOUNT SIMULATION")
    print("=" * 90)

    display_columns = [

        "Discount",

        "Selling Price",

        "Predicted Demand",

        "Expected Units",

        "Revenue",

        "Profit",

        "Profit Margin",

        "Inventory Clearance",

        "Feasible"
    ]

    print(
        results[
            display_columns
        ].round(2).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = find_recommendation(
        results
    )

    print("\n" + "=" * 90)
    print("🏆 BEST BUSINESS SCENARIO")
    print("=" * 90)

    print(
        f"Recommended Discount : "
        f"{recommendation['Discount']:.0f}%"
    )

    print(
        f"Selling Price       : "
        f"₹{recommendation['Selling Price']:.2f}"
    )

    print(
        f"Expected Demand     : "
        f"{recommendation['Predicted Demand']:.0f}"
    )

    print(
        f"Expected Units      : "
        f"{recommendation['Expected Units']:.0f}"
    )

    print(
        f"Expected Revenue    : "
        f"₹{recommendation['Revenue']:,.2f}"
    )

    print(
        f"Expected Profit     : "
        f"₹{recommendation['Profit']:,.2f}"
    )

    print(
        f"Profit Margin       : "
        f"{recommendation['Profit Margin']:.2f}%"
    )

    print(
        f"Inventory Clearance : "
        f"{recommendation['Inventory Clearance']:.2f}%"
    )

    print("\n" + "=" * 90)
    print("SIMULATION COMPLETE")
    print("=" * 90)