# -*- coding: utf-8 -*-
"""
House Rent Prediction Agent
============================
Generates synthetic rental data, trains and compares multiple models
(Linear Regression, Random Forest, XGBoost/Gradient Boosting),
then runs an interactive prediction agent where you can input house
features and get a rent estimate with confidence interval.

Requirements:
    pip install scikit-learn pandas numpy xgboost
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------

def generate_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic house rental data."""
    rng = np.random.default_rng(seed)

    locations = ["Downtown", "Suburbs", "Rural", "City Centre", "Uptown"]
    location_premium = {"Downtown": 1.4, "City Centre": 1.3, "Uptown": 1.15,
                        "Suburbs": 1.0, "Rural": 0.75}

    property_types = ["Apartment", "House", "Studio", "Condo", "Villa"]
    type_premium   = {"Villa": 1.5, "House": 1.2, "Condo": 1.1,
                      "Apartment": 1.0, "Studio": 0.75}

    location     = rng.choice(locations, n_samples)
    prop_type    = rng.choice(property_types, n_samples)
    bedrooms     = rng.integers(1, 6, n_samples)
    bathrooms    = np.clip(rng.integers(1, 4, n_samples), 1, bedrooms)
    area_sqft    = rng.integers(300, 4000, n_samples).astype(float)
    age_years    = rng.integers(0, 40, n_samples).astype(float)
    floor        = rng.integers(0, 30, n_samples).astype(float)
    has_parking  = rng.integers(0, 2, n_samples)
    has_gym      = rng.integers(0, 2, n_samples)
    has_pool     = rng.integers(0, 2, n_samples)
    furnished    = rng.integers(0, 2, n_samples)

    base_rent = (
        300
        + area_sqft * 0.55
        + bedrooms  * 120
        + bathrooms * 80
        - age_years * 5
        + floor     * 8
        + has_parking * 80
        + has_gym     * 60
        + has_pool    * 100
        + furnished   * 150
    )

    loc_factor  = np.array([location_premium[l] for l in location])
    type_factor = np.array([type_premium[t]     for t in prop_type])
    noise       = rng.normal(0, 80, n_samples)
    rent        = np.round(base_rent * loc_factor * type_factor + noise, 2)
    rent        = np.clip(rent, 200, 15000)

    return pd.DataFrame({
        "location":     location,
        "property_type": prop_type,
        "bedrooms":     bedrooms,
        "bathrooms":    bathrooms,
        "area_sqft":    area_sqft,
        "age_years":    age_years,
        "floor":        floor,
        "has_parking":  has_parking,
        "has_gym":      has_gym,
        "has_pool":     has_pool,
        "furnished":    furnished,
        "rent":         rent,
    })


# ---------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------

def preprocess(df: pd.DataFrame):
    """Encode categoricals, scale numerics, return X, y and encoders."""
    df = df.copy()

    le_location = LabelEncoder()
    le_type     = LabelEncoder()
    df["location"]      = le_location.fit_transform(df["location"])
    df["property_type"] = le_type.fit_transform(df["property_type"])

    X = df.drop(columns=["rent"])
    y = df["rent"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le_location, le_type, list(X.columns)


# ---------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# ---------------------------------------------

def train_and_evaluate(X, y):
    """Train three models, print metrics, return the best one."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression":      LinearRegression(),
        "Random Forest":          RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting":      GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    header = f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>7} {'CV R²':>8}"
    print(header)
    print("-" * 60)

    best_model, best_r2 = None, -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds  = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2   = r2_score(y_test, preds)
        cv   = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

        print(f"{name:<25} {mae:>8.1f} {rmse:>8.1f} {r2:>7.3f} {cv:>8.3f}")

        if r2 > best_r2:
            best_r2    = r2
            best_model = (name, model)

    print("-" * 60)
    if best_model is None:
        raise ValueError("No models were evaluated — cannot determine best model.")
    print(f"  v Best model: {best_model[0]}  (R² = {best_r2:.3f})")
    print("=" * 60 + "\n")

    return best_model[1]


# ---------------------------------------------
# 4. INTERACTIVE PREDICTION AGENT
# ---------------------------------------------

LOCATIONS      = ["Downtown", "Suburbs", "Rural", "City Centre", "Uptown"]
PROPERTY_TYPES = ["Apartment", "House", "Studio", "Condo", "Villa"]


def prompt_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            val = int(input(f"  {prompt} [{lo}-{hi}]: "))
            if lo <= val <= hi:
                return val
            print(f"    !  Please enter a number between {lo} and {hi}.")
        except ValueError:
            print("    !  Invalid input — enter a whole number.")


def prompt_choice(prompt: str, choices: list) -> str:
    print(f"\n  {prompt}")
    for i, c in enumerate(choices, 1):
        print(f"    {i}. {c}")
    while True:
        try:
            idx = int(input("  Choice: "))
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            print(f"    !  Enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("    !  Invalid input.")


def prompt_bool(prompt: str) -> int:
    while True:
        ans = input(f"  {prompt} (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return 1
        if ans in ("n", "no"):
            return 0
        print("    !  Please enter y or n.")


def predict_rent(model, scaler, le_location, le_type, feature_names):
    """Collect house features interactively and print rent prediction."""
    print("\n" + "=" * 60)
    print("  HOUSE RENT PREDICTION AGENT")
    print("  Enter the property details below.")
    print("=" * 60)

    location     = prompt_choice("Location:", LOCATIONS)
    prop_type    = prompt_choice("Property type:", PROPERTY_TYPES)
    bedrooms     = prompt_int("Bedrooms",    1, 5)
    bathrooms    = prompt_int("Bathrooms",   1, min(bedrooms, 3))
    area_sqft    = prompt_int("Area (sq ft)", 100, 10000)
    age_years    = prompt_int("Property age (years)", 0, 80)
    floor        = prompt_int("Floor number", 0, 50)
    has_parking  = prompt_bool("Has parking?")
    has_gym      = prompt_bool("Has gym?")
    has_pool     = prompt_bool("Has pool?")
    furnished    = prompt_bool("Furnished?")

    loc_enc  = le_location.transform([location])[0]
    type_enc = le_type.transform([prop_type])[0]

    raw = pd.DataFrame([[loc_enc, type_enc, bedrooms, bathrooms,
                         area_sqft, age_years, floor,
                         has_parking, has_gym, has_pool, furnished]],
                       columns=feature_names)

    scaled = scaler.transform(raw)
    rent   = model.predict(scaled)[0]

    # Rough ±10 % confidence band
    lo_est = rent * 0.90
    hi_est = rent * 1.10

    print("\n" + "-" * 60)
    print(f"  >> {prop_type} in {location}")
    print(f"     {bedrooms} bed  |  {bathrooms} bath  |  {area_sqft} sqft")
    print(f"\n  Estimated Monthly Rent:  ₹{rent:,.0f}")
    print(f"  Typical range:           ₹{lo_est:,.0f}  -  ₹{hi_est:,.0f}")
    print("-" * 60 + "\n")


# ---------------------------------------------
# 5. MAIN
# ---------------------------------------------

def main():
    print("\n  Generating training data …")
    df = generate_data(n_samples=3000)

    print("  Preprocessing …")
    X, y, scaler, le_location, le_type, feature_names = preprocess(df)

    print("  Training models …")
    best_model = train_and_evaluate(X, y)

    while True:
        predict_rent(best_model, scaler, le_location, le_type, feature_names)
        again = input("  Predict another property? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("\n  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
