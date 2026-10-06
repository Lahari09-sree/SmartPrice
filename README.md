?? SmartPrice
Explainable Multi-Objective Retail Pricing & Discount Optimization System
SmartPrice is an AI-powered retail pricing decision-support system that analyzes sales, demand, inventory, competitor pricing, discounts, promotions, and seasonal factors to recommend profitable pricing and discount strategies.
Unlike a conventional sales dashboard, SmartPrice combines machine learning, business constraints, what-if simulation, inventory intelligence, and explainable recommendations to support data-driven pricing decisions.

?? Problem Statement
Retail businesses need to decide:
* What price should a product be sold at?
* How much discount should be offered?
* When should discounts be increased or reduced?
* How can excess inventory be cleared without destroying profit?
* How should competitor prices influence pricing decisions?
* Why was a particular discount recommended?
A discount may increase demand but reduce profit margin.
SmartPrice addresses this trade-off by evaluating multiple pricing scenarios and selecting the best feasible decision.

?? Solution
SmartPrice follows this workflow:
Sales Data + Inventory + Competitor Pricing + Market Conditions
??
Data Preprocessing & Feature Engineering
??
Machine Learning Demand Prediction
??
Multiple Discount Scenario Simulation
??
Business Constraints & Multi-Objective Optimization
??
Optimal Discount Recommendation
??
Explainable + Counterfactual Decision

?? Key Features
1. ?? Sales & Demand Analytics
Analyze:
* Revenue
* Profit
* Units Sold
* Demand
* Discount impact
* Seasonal demand
* Category performance
* Promotional demand lift

2. ?? Inventory Intelligence
SmartPrice evaluates inventory pressure using the relationship between inventory and expected demand.
Products are classified as:
* Stock Risk
* Tight
* Balanced
* Overstocked
* Severe Overstock
This helps identify products that may require inventory clearance strategies.

3. ?? ML-Based Demand Prediction
A Random Forest Regressor predicts expected product demand using features such as:
* Product
* Store
* Category
* Region
* Inventory
* Units Ordered
* Price
* Discount
* Promotion
* Competitor Pricing
* Seasonality
* Weather
* Epidemic indicator
* Calendar features
Model Performance
ModelMAERMSER²Linear Regression23.5630.530.578Random Forest14.4020.280.814The Random Forest model was selected because it achieved the best validation performance.

?? Smart Pricing Optimization
SmartPrice evaluates multiple discount scenarios:
0%, 5%, 10%, 15%, 20%, 25%
For every scenario it estimates:
* Selling price
* Predicted demand
* Expected units sold
* Revenue
* Profit
* Profit margin
* Inventory clearance
* Feasibility
A minimum 10% profit-margin constraint prevents unsafe discount recommendations.

?? What-If Pricing Simulator
Users can simulate different discount levels before making a pricing decision.
For each scenario, the dashboard displays:
* Expected demand
* Expected revenue
* Expected profit
* Profit margin
* Inventory clearance
This allows users to answer questions such as:
"What happens if I increase the discount from 10% to 20%?"

?? Explainable "Why NOT?" Analysis
SmartPrice does not simply provide a recommendation.
It also explains why alternative discounts were rejected.
For example:
Recommended: 0% discount
Why NOT 10%?
Expected profit decreases compared with the recommended scenario.
Why NOT 20%?
Expected profit decreases further.
Why NOT 25%?
The expected profit margin falls below the 10% safety threshold.
This provides counterfactual explanations that make the pricing recommendation easier for business users to understand.

?? Multi-Objective Decision Making
SmartPrice considers multiple business objectives instead of optimizing only for sales.
The decision framework considers:
* Profit maximization
* Demand
* Inventory pressure
* Competitor pricing
* Profit-margin protection
* Inventory clearance
Therefore, the system is designed as a business decision-support system, rather than only a machine-learning prediction model.

??? Project Structure
SmartPrice/
?
??? data/
?   ??? sales_data.csv
?   ??? processed_sales_data.csv
?
??? models/
?   ??? demand_model.pkl
?
??? notebooks/
?
??? src/
?   ??? data_preprocessing.py
?   ??? eda.py
?   ??? demand_model.py
?   ??? pricing_optimizer.py
?   ??? smart_optimizer.py
?   ??? what_if_simulator.py
?
??? dashboard/
?   ??? app.py
?
??? README.md
??? .gitignore

??? Technology Stack
Programming
* Python
Data Analysis
* Pandas
* NumPy
Machine Learning
* Scikit-learn
* Random Forest Regression
Visualization
* Plotly
* Streamlit
Data / Database Skills
* SQL
* Excel
Model Management
* Joblib

?? Dataset
The project uses a retail store inventory and demand dataset containing approximately 76,000 records and multiple retail attributes.
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
The dataset was processed to create additional analytical features such as:
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

?? Business Insights
The analysis identified several useful patterns:
* Promotional periods show higher average demand than non-promotional periods.
* Increasing discounts can increase demand but reduce profit margin.
* Excess inventory can be identified using inventory-to-demand ratios.
* Competitor pricing provides additional pricing context.
* A discount that maximizes demand is not necessarily the discount that maximizes profit.
This demonstrates the importance of balancing demand growth with profitability.

?? How to Run
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SmartPrice
2. Install dependencies
pip install pandas numpy scikit-learn joblib streamlit plotly
3. Run the dashboard
python -m streamlit run dashboard/app.py
The Streamlit application will open in your browser.

?? Use Cases
SmartPrice can support:
* Retail pricing teams
* E-commerce businesses
* Inventory management
* Promotional planning
* Revenue management
* Business analysts
* Data-driven pricing decisions

?? Future Enhancements
Potential future improvements include:
* Time-series demand forecasting
* Real-time competitor price APIs
* SQL database integration
* Automated pricing alerts
* Dynamic pricing based on real-time inventory
* Advanced optimization using Bayesian or evolutionary methods
* Customer-level personalized pricing
* Cloud deployment
* Power BI integration

????? Project Author
Y. Lahari Sree
Integrated M.Tech – Computer Science Engineering, Business Analytics
VIT Chennai
Core Areas
Python • SQL • Data Analytics • Machine Learning • Business Analytics • Dashboard Development

? Project Highlights
ML-powered demand prediction + business-rule optimization + what-if simulation + explainable counterfactual pricing
SmartPrice demonstrates how machine learning can be combined with business analytics and decision-making constraints to solve a practical retail pricing problem.

### Step 2 — Save it

Save as:

```text
C:\Users\lahar\Downloads\SmartPrice\README.md
Important: Don't put your GitHub URL yet. Leave this line:
git clone <YOUR_GITHUB_REPOSITORY_URL>
We'll replace it after you create the repository.
Step 3 — Check it
Open PowerShell:
cd C:\Users\lahar\Downloads\SmartPrice
Then run:
dir
You should see:
data
dashboard
models
notebooks
src
README.md
Once you confirm README.md is there, we'll create the .gitignore and then move to GitHub upload.

