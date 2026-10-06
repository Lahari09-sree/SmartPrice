# SmartPrice — Explainable Multi-Objective Retail Pricing & Discount Optimization System

SmartPrice is an AI-powered retail pricing decision-support system that combines **machine learning, business constraints, inventory intelligence, competitor pricing, what-if simulation, and explainable recommendations** to identify profitable pricing and discount strategies.

Instead of optimizing only for sales or demand, SmartPrice evaluates multiple business factors such as **profit, demand, inventory pressure, competitor pricing, and profit-margin protection** before recommending a discount.

---

## Problem Statement

Retail businesses need to continuously answer questions such as:

* What price should a product be sold at?
* How much discount should be offered?
* Will increasing the discount improve demand enough to justify the lost margin?
* How can excess inventory be cleared without significantly reducing profitability?
* How should competitor pricing influence the decision?
* Why was a particular discount recommended?
* Why were alternative discounts rejected?

A discount may increase demand while simultaneously reducing profit. SmartPrice addresses this trade-off through **multi-scenario pricing analysis and business constraints**.

---

##  Solution

SmartPrice follows this workflow:

```text
Sales + Inventory + Competitor + Market Data
                    ↓
        Data Preprocessing & EDA
                    ↓
          Feature Engineering
                    ↓
        ML-Based Demand Prediction
                    ↓
       Multiple Discount Scenarios
                    ↓
      Business Constraints & Scoring
                    ↓
        Optimal Pricing Decision
                    ↓
     Explainable "Why NOT?" Analysis
                    ↓
          Business Dashboard
```

---

##  Key Features

### 1. Sales & Demand Analytics

The dashboard analyzes:

* Revenue
* Estimated profit
* Units sold
* Demand
* Discount impact
* Seasonal demand
* Category performance
* Promotional demand lift

---

### 2. Inventory Intelligence

SmartPrice evaluates inventory pressure using the relationship between available inventory and expected demand.

Products are classified into:

| Inventory Status | Meaning                                  |
| ---------------- | ---------------------------------------- |
| Stock Risk       | Inventory may be insufficient            |
| Tight            | Limited inventory                        |
| Balanced         | Healthy inventory level                  |
| Overstocked      | Inventory is higher than expected demand |
| Severe Overstock | Significant excess inventory             |

This allows pricing decisions to consider both **profitability and inventory clearance**.

---

### 3. Machine Learning Demand Prediction

A **Random Forest Regressor** predicts expected demand using features such as:

* Product ID
* Store ID
* Category
* Region
* Inventory level
* Units ordered
* Price
* Discount
* Promotion
* Competitor pricing
* Seasonality
* Weather condition
* Epidemic indicator
* Calendar features

#### Model Performance

| Model             |       MAE |      RMSE |        R² |
| ----------------- | --------: | --------: | --------: |
| Linear Regression |     23.56 |     30.53 |     0.578 |
| **Random Forest** | **14.40** | **20.28** | **0.814** |

The Random Forest model was selected because it achieved the strongest validation performance.

---

## Smart Pricing Optimization

SmartPrice evaluates multiple discount scenarios:

```text
0% → 5% → 10% → 15% → 20% → 25%
```

For every scenario, the system estimates:

* Selling price
* Predicted demand
* Expected units sold
* Revenue
* Profit
* Profit margin
* Inventory clearance
* Feasibility

A **minimum 10% profit-margin constraint** prevents the system from recommending discounts that would violate the project's profitability safety rule.

The system therefore searches for a discount that provides the best business outcome rather than simply maximizing demand.

---

##  Explainable "Why NOT?" Analysis

SmartPrice does not only answer:

> **"What discount should I choose?"**

It also answers:

> **"Why shouldn't I choose the other discounts?"**

For example:

```text
Recommended Discount: 0%

Why NOT 10%?
Expected profit is lower than the recommended scenario.

Why NOT 20%?
Expected profit decreases further.

Why NOT 25%?
Expected profit margin falls below the 10% minimum threshold.
```

This provides a **counterfactual explanation** that makes the recommendation easier for business users to understand.

---

##  What-If Pricing Simulator

Users can test different discount scenarios before making a pricing decision.

The simulator compares:

* Expected demand
* Revenue
* Profit
* Profit margin
* Inventory clearance

Example business question:

> **What happens if the discount is increased from 10% to 20%?**

The dashboard allows users to compare the scenarios visually before selecting a pricing strategy.

---

## Multi-Objective Decision Making

SmartPrice considers multiple objectives instead of optimizing only for sales.

The decision framework considers:

* **Profit maximization**
* **Demand**
* **Inventory pressure**
* **Competitor pricing**
* **Profit-margin protection**
* **Inventory clearance**

This makes SmartPrice a **business decision-support system**, rather than simply a demand-prediction model.

---

## Business Insights

The analysis identified several useful patterns:

* Promotional periods show higher average demand than non-promotional periods.
* Increasing discounts can increase demand while reducing profit margin.
* Excess inventory can be identified using inventory-to-demand relationships.
* Competitor pricing provides additional pricing context.
* The discount that maximizes demand is not necessarily the discount that maximizes profit.
* Pricing decisions should balance demand growth with profitability and inventory conditions.

---

##  Project Structure

```text
SmartPrice/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── sales_data.csv
│   └── processed_sales_data.csv
│
├── models/
│   └── demand_model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── demand_model.py
│   ├── pricing_optimizer.py
│   ├── smart_optimizer.py
│   └── what_if_simulator.py
│
├── .gitignore
└── README.md
```

> **Note:** `demand_model.pkl` is excluded from the GitHub repository because the trained model file is larger than GitHub's individual file-size limit. The model can be regenerated using the training code in `src/demand_model.py`.

---

##  Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest Regression
* Joblib

### Visualization & Dashboard

* Streamlit
* Plotly

### Business Analytics

* SQL
* Excel

---

##  Dataset

The project uses a retail inventory and demand forecasting dataset containing approximately **76,000 records**.

Important variables include:

* Date
* Store ID
* Product ID
* Category
* Region
* Inventory Level
* Units Sold
* Units Ordered
* Price
* Discount
* Promotion
* Competitor Pricing
* Seasonality
* Weather Condition
* Demand

Additional analytical features were created during preprocessing, including:

* Discounted Price
* Competitor Price Difference
* Competitor Price Gap %
* Inventory-Demand Ratio
* Inventory Pressure
* Estimated Cost
* Revenue
* Estimated Profit
* Profit Margin
* Calendar features

---

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Lahari09-sree/SmartPrice.git
cd SmartPrice
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib streamlit plotly
```

### 3. Generate the demand model

Run the model-training script if the trained model is not available locally:

```bash
python src/demand_model.py
```

This generates the trained model inside the `models/` directory.

### 4. Start the Streamlit dashboard

```bash
python -m streamlit run dashboard/app.py
```

The application will open in your browser.

---

##  Use Cases

SmartPrice can support:

* Retail pricing teams
* E-commerce businesses
* Inventory management
* Promotional planning
* Revenue management
* Business analysts
* Pricing analysts
* Data-driven retail decision making

---

##  Future Enhancements

Potential future improvements include:

* Time-series demand forecasting
* Real-time competitor price APIs
* SQL database integration
* Automated pricing alerts
* Dynamic pricing using real-time inventory
* Advanced optimization using Bayesian or evolutionary methods
* Customer-level personalized pricing
* Cloud deployment
* Power BI integration

---

##  Author

**Y. Lahari Sree**

Integrated M.Tech — Computer Science Engineering, Business Analytics
**VIT Chennai**

### Core Areas

`Python` • `SQL` • `Data Analytics` • `Machine Learning` • `Business Analytics` • `Dashboard Development`

---

##  Project Highlights

**ML Demand Prediction**
+
**Multi-Objective Pricing Optimization**
+
**Inventory Intelligence**
+
**What-If Simulation**
+
**Explainable Counterfactual Analysis**
+
**Interactive Streamlit Dashboard**

SmartPrice demonstrates how **machine learning can be combined with business analytics and decision-making constraints** to solve a practical retail pricing problem.
