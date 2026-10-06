import os
import joblib
import pandas as pd
import numpy as np

# ============================================================
# SMARTPRICE - MULTI-OBJECTIVE PRICING OPTIMIZER
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed_sales_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "demand_model.pkl")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# ============================================================
# BUSINESS SETTINGS
# ============================================================

COST_RATIO = 0.70
MIN_PROFIT_MARGIN = 10.0

DISCOUNT_OPTIONS = [0, 5, 10, 15, 20, 25]


# ============================================================
# INVENTORY STATUS
# ============================================================

def get_inventory_status(inventory_ratio):

    if inventory_ratio < 1:
        return "Stock Risk"

    elif inventory_ratio < 2:
        return "Tight"

    elif inventory_ratio < 3:
        return "Balanced"

    elif inventory_ratio < 4:
        return "Overstocked"

    else:
        return "Severe Overstock"


# ============================================================
# COMPETITOR POSITION
# ============================================================

def get_competitor_position(price, competitor_price):

    difference = price - competitor_price

    if difference < -2:
        return "Cheaper Than Competitor"

    elif difference > 2:
        return "More Expensive Than Competitor"

    else:
        return "Similar to Competitor"


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
# SMART OPTIMIZER
# ============================================================

def optimize_product(row):

    current_price = float(row["Price"])
    inventory = float(row["Inventory Level"])
    competitor_price = float(row["Competitor Pricing"])

    # Inventory pressure
    predicted_demand_current = predict_demand(row, row["Discount"])

    inventory_ratio = inventory / max(predicted_demand_current, 1)

    inventory_status = get_inventory_status(inventory_ratio)

    competitor_position = get_competitor_position(
        current_price,
        competitor_price
    )

    scenarios = []

    for discount in DISCOUNT_OPTIONS:

        selling_price = current_price * (1 - discount / 100)

        predicted_demand = predict_demand(row, discount)

        expected_units = min(
            predicted_demand,
            inventory
        )

        revenue = expected_units * selling_price

        cost_per_unit = current_price * COST_RATIO

        total_cost = expected_units * cost_per_unit

        profit = revenue - total_cost

        profit_margin = (
            profit / revenue * 100
            if revenue > 0
            else 0
        )

        inventory_clearance = (
            expected_units / inventory * 100
            if inventory > 0
            else 0
        )

        # ----------------------------------------------------
        # Margin constraint
        # ----------------------------------------------------

        margin_score = min(
            profit_margin / MIN_PROFIT_MARGIN,
            1
        )

        # ----------------------------------------------------
        # Profit score
        # ----------------------------------------------------

        profit_score = profit

        # ----------------------------------------------------
        # Inventory score
        # ----------------------------------------------------

        if inventory_status == "Severe Overstock":
            inventory_score = inventory_clearance * 1.5

        elif inventory_status == "Overstocked":
            inventory_score = inventory_clearance * 1.2

        elif inventory_status == "Balanced":
            inventory_score = inventory_clearance

        elif inventory_status == "Tight":
            inventory_score = -inventory_clearance

        else:
            inventory_score = -inventory_clearance * 1.5

        # ----------------------------------------------------
        # Competitor score
        # ----------------------------------------------------

        price_difference_percent = (
            (selling_price - competitor_price)
            / competitor_price
        ) * 100

        if price_difference_percent <= 0:
            competitor_score = 10
        elif price_difference_percent <= 5:
            competitor_score = 5
        else:
            competitor_score = 0

        # ----------------------------------------------------
        # Feasibility
        # ----------------------------------------------------

        feasible = profit_margin >= MIN_PROFIT_MARGIN

        # ----------------------------------------------------
        # Normalize profit later
        # ----------------------------------------------------

        scenarios.append({
            "Discount": discount,
            "Selling Price": selling_price,
            "Predicted Demand": predicted_demand,
            "Expected Units": expected_units,
            "Revenue": revenue,
            "Profit": profit,
            "Profit Margin": profit_margin,
            "Inventory Clearance": inventory_clearance,
            "Competitor Gap %": price_difference_percent,
            "Competitor Score": competitor_score,
            "Margin Score": margin_score,
            "Inventory Score": inventory_score,
            "Feasible": feasible
        })

    results = pd.DataFrame(scenarios)

    # ========================================================
    # NORMALIZE PROFIT
    # ========================================================

    max_profit = results["Profit"].max()

    if max_profit > 0:
        results["Profit Score"] = (
            results["Profit"] / max_profit
        ) * 100
    else:
        results["Profit Score"] = 0

    # ========================================================
    # MULTI-OBJECTIVE SMART SCORE
    # ========================================================

    results["Smart Score"] = (
        results["Profit Score"] * 0.50
        + results["Inventory Score"].clip(-100, 100) * 0.20
        + results["Competitor Score"] * 0.10
        + results["Margin Score"] * 100 * 0.20
    )

    # ========================================================
    # APPLY BUSINESS CONSTRAINT
    # ========================================================

    feasible_results = results[
        results["Feasible"] == True
    ]

    if len(feasible_results) > 0:

        best_index = feasible_results[
            "Smart Score"
        ].idxmax()

    else:

        best_index = results[
            "Profit Margin"
        ].idxmax()

    recommendation = results.loc[best_index]

    return (
        results,
        recommendation,
        inventory_status,
        inventory_ratio,
        competitor_position
    )


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def generate_explanation(
    recommendation,
    inventory_status,
    competitor_position
):

    discount = recommendation["Discount"]
    margin = recommendation["Profit Margin"]
    profit = recommendation["Profit"]

    reasons = []

    if inventory_status == "Severe Overstock":
        reasons.append(
            "inventory is severely overstocked"
        )

    elif inventory_status == "Overstocked":
        reasons.append(
            "inventory pressure is high"
        )

    elif inventory_status == "Tight":
        reasons.append(
            "inventory is relatively tight"
        )

    elif inventory_status == "Stock Risk":
        reasons.append(
            "inventory is limited"
        )

    else:
        reasons.append(
            "inventory is relatively balanced"
        )

    if competitor_position == "More Expensive Than Competitor":
        reasons.append(
            "the product is priced above the competitor"
        )

    elif competitor_position == "Cheaper Than Competitor":
        reasons.append(
            "the product is already cheaper than the competitor"
        )

    else:
        reasons.append(
            "the price is close to the competitor"
        )

    reasons.append(
        f"the recommended discount maintains a {margin:.2f}% profit margin"
    )

    reasons.append(
        f"while generating approximately ₹{profit:,.2f} expected profit"
    )

    return (
        f"Recommended {discount}% discount because "
        + ", ".join(reasons)
        + "."
    )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 90)
    print("SMARTPRICE - MULTI-OBJECTIVE PRICING OPTIMIZATION ENGINE")
    print("=" * 90)

    # Select sample product
    sample_row = df.iloc[0]

    (
        results,
        recommendation,
        inventory_status,
        inventory_ratio,
        competitor_position
    ) = optimize_product(sample_row)

    # --------------------------------------------------------
    # PRODUCT INFORMATION
    # --------------------------------------------------------

    print("\nPRODUCT INFORMATION")
    print("-" * 90)

    print(f"Product ID          : {sample_row['Product ID']}")
    print(f"Store ID            : {sample_row['Store ID']}")
    print(f"Category            : {sample_row['Category']}")
    print(f"Current Price       : ₹{sample_row['Price']:.2f}")
    print(f"Inventory           : {sample_row['Inventory Level']:.0f}")
    print(f"Competitor Price    : ₹{sample_row['Competitor Pricing']:.2f}")
    print(f"Current Discount    : {sample_row['Discount']:.0f}%")

    print(f"\nInventory Ratio     : {inventory_ratio:.2f}")
    print(f"Inventory Status    : {inventory_status}")
    print(f"Competitor Position : {competitor_position}")

    # --------------------------------------------------------
    # SCENARIOS
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("MULTI-OBJECTIVE DISCOUNT SCENARIOS")
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
        "Smart Score",
        "Feasible"
    ]

    print(
        results[display_columns].round(2).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("🏆 SMARTPRICE RECOMMENDATION")
    print("=" * 90)

    print(
        f"Recommended Discount : "
        f"{recommendation['Discount']:.0f}%"
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
        f"Inventory Clearance  : "
        f"{recommendation['Inventory Clearance']:.2f}%"
    )

    print(
        f"Smart Score          : "
        f"{recommendation['Smart Score']:.2f}"
    )

    explanation = generate_explanation(
        recommendation,
        inventory_status,
        competitor_position
    )

    print("\n💡 WHY THIS RECOMMENDATION?")
    print(explanation)

    print("\n" + "=" * 90)
    print("SMARTPRICE OPTIMIZATION COMPLETE")
    print("=" * 90)