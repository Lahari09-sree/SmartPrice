import os
import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# SMARTPRICE - RETAIL PRICING INTELLIGENCE DASHBOARD
# ============================================================

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from smart_optimizer import optimize_product


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartPrice | Retail Pricing Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.metric-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e6e9ef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.hero {
    padding: 25px 30px;
    border-radius: 16px;
    background: linear-gradient(135deg, #111827, #374151);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.9;
}

.recommendation {
    padding: 25px;
    border-radius: 16px;
    background: #ffffff;
    border: 2px solid #d1d5db;
    margin-top: 10px;
}

.small-text {
    color: #6b7280;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

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
# HELPER FUNCTIONS
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

    return max(
        0,
        float(model.predict(features)[0])
    )


def simulate_discounts(row):

    options = [0, 5, 10, 15, 20, 25]

    scenarios = []

    cost_ratio = 0.70

    for discount in options:

        price = float(row["Price"])

        selling_price = (
            price * (1 - discount / 100)
        )

        demand = predict_demand(
            row,
            discount
        )

        inventory = float(
            row["Inventory Level"]
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
            price * cost_ratio
        )

        profit = (
            revenue -
            expected_units * cost_per_unit
        )

        margin = (
            profit / revenue * 100
            if revenue > 0
            else 0
        )

        clearance = (
            expected_units /
            inventory * 100
            if inventory > 0
            else 0
        )

        scenarios.append({
            "Discount": discount,
            "Selling Price": selling_price,
            "Predicted Demand": demand,
            "Expected Units": expected_units,
            "Revenue": revenue,
            "Profit": profit,
            "Profit Margin": margin,
            "Inventory Clearance": clearance,
            "Feasible": margin >= 10
        })

    return pd.DataFrame(scenarios)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💰 SmartPrice")

st.sidebar.markdown(
    "### Retail Pricing Intelligence"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Executive Overview",
        "📊 Sales & Demand",
        "📦 Inventory Intelligence",
        "💰 Smart Pricing",
        "🔄 What-If Simulator"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-powered pricing decision support"
)

st.sidebar.caption(
    "Python • SQL • ML • Streamlit • Plotly"
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>💰 SmartPrice</h1>

<p>
Explainable Multi-Objective Retail Pricing & Discount Optimization
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.header("Executive Overview")

    st.write(
        "SmartPrice analyzes sales, demand, inventory, "
        "competitor pricing and discounts to support "
        "better retail pricing decisions."
    )

    st.markdown("---")

    # KPIs

    total_revenue = df["Actual_Revenue"].sum()

    total_profit = df["Estimated_Profit"].sum()

    total_units = df["Units Sold"].sum()

    total_demand = df["Demand"].sum()

    avg_discount = df["Discount"].mean()

    avg_margin = df["Profit_Margin_Percent"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Total Revenue",
        f"₹{total_revenue/1e7:.2f} Cr"
    )

    c2.metric(
        "Estimated Profit",
        f"₹{total_profit/1e7:.2f} Cr"
    )

    c3.metric(
        "Units Sold",
        f"{total_units:,.0f}"
    )

    c4.metric(
        "Total Demand",
        f"{total_demand:,.0f}"
    )

    c5.metric(
        "Avg Discount",
        f"{avg_discount:.2f}%"
    )

    c6.metric(
        "Avg Margin",
        f"{avg_margin:.2f}%"
    )

    st.markdown("---")

    # Charts

    col1, col2 = st.columns(2)

    with col1:

        category_sales = (
            df.groupby("Category")["Actual_Revenue"]
            .sum()
            .reset_index()
            .sort_values(
                "Actual_Revenue",
                ascending=False
            )
        )

        fig = px.bar(
            category_sales,
            x="Category",
            y="Actual_Revenue",
            title="Revenue by Category",
            text_auto=".2s"
        )

        fig.update_layout(
            yaxis_title="Revenue (₹)",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        category_demand = (
            df.groupby("Category")["Demand"]
            .sum()
            .reset_index()
            .sort_values(
                "Demand",
                ascending=False
            )
        )

        fig = px.bar(
            category_demand,
            x="Category",
            y="Demand",
            title="Demand by Category",
            text_auto=".2s"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Business insights

    st.subheader("💡 Key Business Insights")

    promotion_demand = (
        df.groupby("Promotion")["Demand"]
        .mean()
    )

    normal_demand = promotion_demand.get(
        0,
        0
    )

    promoted_demand = promotion_demand.get(
        1,
        0
    )

    if normal_demand > 0:

        promotion_lift = (
            (promoted_demand - normal_demand)
            / normal_demand
        ) * 100

    else:

        promotion_lift = 0

    st.info(
        f"Promotional periods show approximately "
        f"**{promotion_lift:.1f}% higher average demand** "
        f"than non-promotional periods in this dataset."
    )


# ============================================================
# SALES & DEMAND
# ============================================================

elif page == "📊 Sales & Demand":

    st.header("📊 Sales & Demand Analytics")

    col1, col2 = st.columns(2)

    with col1:

        discount_demand = (
            df.groupby("Discount")["Demand"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            discount_demand,
            x="Discount",
            y="Demand",
            markers=True,
            title="Discount vs Average Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        discount_profit = (
            df.groupby("Discount")["Estimated_Profit"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            discount_profit,
            x="Discount",
            y="Estimated_Profit",
            title="Discount vs Estimated Profit"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Seasonal Demand")

    seasonal = (
        df.groupby("Seasonality")
        .agg(
            Demand=("Demand", "mean"),
            Units=("Units Sold", "sum"),
            Revenue=("Actual_Revenue", "sum")
        )
        .reset_index()
    )

    fig = px.bar(
        seasonal,
        x="Seasonality",
        y="Demand",
        title="Average Demand by Season",
        text_auto=".1f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Category Performance")

    category = (
        df.groupby("Category")
        .agg(
            Demand=("Demand", "sum"),
            Units=("Units Sold", "sum"),
            Revenue=("Actual_Revenue", "sum"),
            Profit=("Estimated_Profit", "sum"),
            Avg_Discount=("Discount", "mean")
        )
        .reset_index()
    )

    st.dataframe(
        category.style.format({
            "Demand": "{:,.0f}",
            "Units": "{:,.0f}",
            "Revenue": "₹{:,.0f}",
            "Profit": "₹{:,.0f}",
            "Avg_Discount": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INVENTORY INTELLIGENCE
# ============================================================

elif page == "📦 Inventory Intelligence":

    st.header("📦 Inventory Intelligence")

    inventory_summary = (
        df.groupby("Product ID")
        .agg(
            Avg_Inventory=("Inventory Level", "mean"),
            Avg_Demand=("Demand", "mean"),
            Units_Sold=("Units Sold", "sum"),
            Revenue=("Actual_Revenue", "sum")
        )
        .reset_index()
    )

    inventory_summary["Inventory Ratio"] = (
        inventory_summary["Avg_Inventory"]
        / inventory_summary["Avg_Demand"].replace(
            0,
            np.nan
        )
    )

    inventory_summary["Inventory Status"] = pd.cut(
        inventory_summary["Inventory Ratio"],
        bins=[
            -np.inf,
            1,
            2,
            3,
            4,
            np.inf
        ],
        labels=[
            "Stock Risk",
            "Tight",
            "Balanced",
            "Overstocked",
            "Severe Overstock"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            inventory_summary,
            x="Avg_Demand",
            y="Avg_Inventory",
            size="Revenue",
            hover_name="Product ID",
            color="Inventory Status",
            title="Inventory vs Average Demand"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        status_count = (
            inventory_summary[
                "Inventory Status"
            ]
            .value_counts()
            .reset_index()
        )

        status_count.columns = [
            "Status",
            "Products"
        ]

        fig = px.bar(
            status_count,
            x="Status",
            y="Products",
            title="Products by Inventory Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Product Inventory Intelligence")

    st.dataframe(
        inventory_summary.sort_values(
            "Inventory Ratio",
            ascending=False
        ).style.format({
            "Avg_Inventory": "{:.1f}",
            "Avg_Demand": "{:.1f}",
            "Units_Sold": "{:,.0f}",
            "Revenue": "₹{:,.0f}",
            "Inventory Ratio": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SMART PRICING
# ============================================================

elif page == "💰 Smart Pricing":

    st.header("💰 Smart Pricing Recommendation")

    products = sorted(
        df["Product ID"].unique()
    )

    selected_product = st.selectbox(
        "Select Product",
        products
    )

    product_rows = df[
        df["Product ID"] == selected_product
    ]

    stores = sorted(
        product_rows["Store ID"].unique()
    )

    selected_store = st.selectbox(
        "Select Store",
        stores
    )

    matching_rows = product_rows[
        product_rows["Store ID"] == selected_store
    ]

    selected_row = matching_rows.iloc[0]

    (
        results,
        recommendation,
        inventory_status,
        inventory_ratio,
        competitor_position
    ) = optimize_product(selected_row)

    # Product details

    st.markdown("### Product Context")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Price",
        f"₹{selected_row['Price']:.2f}"
    )

    c2.metric(
        "Inventory",
        f"{selected_row['Inventory Level']:.0f}"
    )

    c3.metric(
        "Competitor Price",
        f"₹{selected_row['Competitor Pricing']:.2f}"
    )

    c4.metric(
        "Current Discount",
        f"{selected_row['Discount']:.0f}%"
    )

    st.markdown("---")

    # Recommendation

    st.markdown(
        '<div class="recommendation">',
        unsafe_allow_html=True
    )

    st.subheader("🏆 SmartPrice Recommendation")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Recommended Discount",
        f"{recommendation['Discount']:.0f}%"
    )

    r2.metric(
        "Recommended Price",
        f"₹{recommendation['Selling Price']:.2f}"
    )

    r3.metric(
        "Expected Profit",
        f"₹{recommendation['Profit']:,.2f}"
    )

    r4.metric(
        "Profit Margin",
        f"{recommendation['Profit Margin']:.2f}%"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # Context

    st.subheader("Business Context")

    c1, c2, c3 = st.columns(3)

    c1.info(
        f"Inventory Status\n\n"
        f"**{inventory_status}**"
    )

    c2.info(
        f"Inventory Ratio\n\n"
        f"**{inventory_ratio:.2f}**"
    )

    c3.info(
        f"Competitor Position\n\n"
        f"**{competitor_position}**"
    )

    # Explanation

    st.subheader("💡 Why this recommendation?")

    reasons = []

    if inventory_status == "Overstocked":
        reasons.append(
            "Inventory pressure is high, so a discount "
            "can help improve inventory clearance."
        )

    elif inventory_status == "Severe Overstock":
        reasons.append(
            "Inventory is severely high, making "
            "inventory clearance an important objective."
        )

    elif inventory_status == "Tight":
        reasons.append(
            "Inventory is relatively tight, so SmartPrice "
            "prioritizes protecting margin."
        )

    elif inventory_status == "Stock Risk":
        reasons.append(
            "Inventory is limited, so aggressive discounting "
            "is avoided."
        )

    else:
        reasons.append(
            "Inventory is relatively balanced."
        )

    if competitor_position == "Cheaper Than Competitor":
        reasons.append(
            "The current price is already below the competitor."
        )

    elif competitor_position == "More Expensive Than Competitor":
        reasons.append(
            "The product is priced above the competitor, "
            "so competitive pricing is considered."
        )

    for reason in reasons:
        st.write("•", reason)

    st.write(
        f"• Expected profit at the recommended scenario is "
        f"₹{recommendation['Profit']:,.2f}."
    )

    st.write(
        f"• Expected profit margin is "
        f"{recommendation['Profit Margin']:.2f}%."
    )
    # ========================================================
    # WHY NOT? COUNTERFACTUAL ANALYSIS
    # ========================================================

    st.subheader("🔍 Why NOT the Other Discounts?")

    recommended_discount = recommendation["Discount"]
    recommended_profit = recommendation["Profit"]

    counterfactual = results.copy()

    counterfactual["Profit Difference"] = (
        counterfactual["Profit"] - recommended_profit
    )

    counterfactual["Profit Difference"] = (
        counterfactual["Profit Difference"].round(2)
    )

    counterfactual["Decision"] = counterfactual.apply(
        lambda x: (
            "✅ Recommended"
            if x["Discount"] == recommended_discount
            else (
                "❌ Below minimum margin"
                if not x["Feasible"]
                else "Alternative"
            )
        ),
        axis=1
    )

    counterfactual["Why Not?"] = counterfactual.apply(
        lambda x: (
            "Best feasible profit"
            if x["Discount"] == recommended_discount
            else (
                "Profit margin falls below the 10% safety threshold"
                if not x["Feasible"]
                else (
                    f"Expected profit decreases by "
                    f"₹{abs(x['Profit Difference']):,.2f}"
                )
            )
        ),
        axis=1
    )

    display_columns = [
        "Discount",
        "Selling Price",
        "Predicted Demand",
        "Profit",
        "Profit Margin",
        "Profit Difference",
        "Decision",
        "Why Not?"
    ]

    st.dataframe(
        counterfactual[display_columns].style.format({
            "Discount": "{:.0f}%",
            "Selling Price": "₹{:.2f}",
            "Predicted Demand": "{:.1f}",
            "Profit": "₹{:,.2f}",
            "Profit Margin": "{:.2f}%",
            "Profit Difference": "₹{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Natural-language explanation
    # --------------------------------------------------------

    alternatives = counterfactual[
        counterfactual["Discount"] != recommended_discount
    ]

    st.markdown("### 💡 Counterfactual Explanation")

    st.success(
        f"SmartPrice recommends **{recommended_discount:.0f}% discount** "
        f"because it produces the highest expected profit among "
        f"scenarios satisfying the **10% minimum profit-margin constraint**."
    )

    for _, scenario in alternatives.iterrows():

        discount = scenario["Discount"]
        difference = scenario["Profit Difference"]

        if not scenario["Feasible"]:

            st.warning(
                f"**Why NOT {discount:.0f}%?** "
                f"The expected profit margin is only "
                f"**{scenario['Profit Margin']:.2f}%**, "
                f"which violates the 10% minimum margin requirement."
            )

        else:

            st.info(
                f"**Why NOT {discount:.0f}%?** "
                f"Expected profit decreases by "
                f"**₹{abs(difference):,.2f}** compared with "
                f"the recommended scenario."
            )

    # Scenario chart

    st.subheader("Discount Scenario Comparison")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=results["Discount"],
            y=results["Profit"],
            name="Profit"
        )
    )

    fig.update_layout(
        xaxis_title="Discount (%)",
        yaxis_title="Expected Profit (₹)",
        title="Expected Profit by Discount"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        results.round(2),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

elif page == "🔄 What-If Simulator":

    st.header("🔄 What-If Pricing Simulator")

    st.write(
        "Test different discount levels and compare their "
        "expected demand, revenue, profit and margin."
    )

    products = sorted(
        df["Product ID"].unique()
    )

    selected_product = st.selectbox(
        "Select Product",
        products,
        key="what_if_product"
    )

    product_rows = df[
        df["Product ID"] == selected_product
    ]

    stores = sorted(
        product_rows["Store ID"].unique()
    )

    selected_store = st.selectbox(
        "Select Store",
        stores,
        key="what_if_store"
    )

    selected_row = product_rows[
        product_rows["Store ID"] == selected_store
    ].iloc[0]

    results = simulate_discounts(
        selected_row
    )

    feasible = results[
        results["Feasible"] == True
    ]

    if len(feasible) > 0:

        best = feasible.loc[
            feasible["Profit"].idxmax()
        ]

    else:

        best = results.loc[
            results["Profit Margin"].idxmax()
        ]

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Recommended Discount",
        f"{best['Discount']:.0f}%"
    )

    c2.metric(
        "Expected Demand",
        f"{best['Predicted Demand']:.0f}"
    )

    c3.metric(
        "Expected Profit",
        f"₹{best['Profit']:,.2f}"
    )

    c4.metric(
        "Profit Margin",
        f"{best['Profit Margin']:.2f}%"
    )

    st.markdown("---")

    # Profit chart

    fig = px.bar(
        results,
        x="Discount",
        y="Profit",
        title="Expected Profit by Discount",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Demand chart

    fig = px.line(
        results,
        x="Discount",
        y="Predicted Demand",
        markers=True,
        title="Expected Demand by Discount"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Margin chart

    fig = px.line(
        results,
        x="Discount",
        y="Profit Margin",
        markers=True,
        title="Profit Margin by Discount"
    )

    fig.add_hline(
        y=10,
        line_dash="dash",
        annotation_text="Minimum Margin"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Scenario Table")

    st.dataframe(
        results.style.format({
            "Selling Price": "₹{:.2f}",
            "Predicted Demand": "{:.1f}",
            "Expected Units": "{:.1f}",
            "Revenue": "₹{:,.2f}",
            "Profit": "₹{:,.2f}",
            "Profit Margin": "{:.2f}%",
            "Inventory Clearance": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"SmartPrice recommends a "
        f"**{best['Discount']:.0f}% discount** "
        f"with expected profit of "
        f"**₹{best['Profit']:,.2f}**."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SmartPrice | Explainable Multi-Objective Retail "
    "Pricing & Discount Optimization System"
)

st.caption(
    "Built with Python • Pandas • Scikit-learn • "
    "Streamlit • Plotly"
)