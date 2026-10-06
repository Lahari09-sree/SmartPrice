import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed_sales_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SMARTPRICE - DEMAND PREDICTION MODEL")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)


# ============================================================
# FEATURES
# ============================================================

# IMPORTANT:
# Units Sold is deliberately excluded to avoid data leakage.

features = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Weather Condition",
    "Promotion",
    "Competitor Pricing",
    "Seasonality",
    "Epidemic",
    "Year",
    "Month",
    "DayOfWeek",
    "IsWeekend"
]

target = "Demand"

X = df[features]
y = df[target]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]

numeric_features = [
    "Inventory Level",
    "Units Ordered",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
    "Year",
    "Month",
    "DayOfWeek",
    "IsWeekend"
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining records: {len(X_train):,}")
print(f"Testing records : {len(X_test):,}")


# ============================================================
# MODELS
# ============================================================

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)


random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=150,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# LINEAR REGRESSION
# ============================================================

print("\nTraining Linear Regression...")

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("Training Random Forest...")

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print("\nLinear Regression")
print(f"MAE  : {linear_mae:.3f}")
print(f"RMSE : {linear_rmse:.3f}")
print(f"R²   : {linear_r2:.4f}")

print("\nRandom Forest")
print(f"MAE  : {rf_mae:.3f}")
print(f"RMSE : {rf_rmse:.3f}")
print(f"R²   : {rf_r2:.4f}")


# ============================================================
# SELECT BEST MODEL
# ============================================================

if rf_r2 >= linear_r2:
    best_model = random_forest_model
    best_model_name = "Random Forest"
else:
    best_model = linear_model
    best_model_name = "Linear Regression"


print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print("=" * 70)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "demand_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print(f"\nModel saved to:")
print(model_path)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

comparison = pd.DataFrame(
    {
        "Actual Demand": y_test.iloc[:10].values,
        "Predicted Demand": np.round(
            best_model.predict(X_test.iloc[:10]),
            2
        )
    }
)

print("\nSample Predictions:")
print(comparison.to_string(index=False))

print("\nDemand prediction model completed successfully.")