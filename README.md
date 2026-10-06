# House Rent Predictor

A machine learning-based command-line tool that estimates monthly house rent based on property features. It generates synthetic rental data, trains and compares three regression models, then launches an interactive agent where you enter property details and receive an estimated monthly rent with a confidence range.

---

## What It Does

1. **Generates** 3,000 synthetic rental records with realistic pricing based on location, property type, size, amenities, and more.
2. **Trains and compares** three models — Linear Regression, Random Forest, and Gradient Boosting — printing MAE, RMSE, R², and cross-validated R² for each.
3. **Selects the best model** automatically (highest R² on the test set).
4. **Runs an interactive agent** that prompts you for property details and outputs an estimated monthly rent plus a ±10% typical price range.

---

## How to Run

```bash
python house_rent_predictor.py
```

Follow the on-screen prompts to enter property details. You can predict multiple properties in one session.

**Example session:**

```
  Generating training data …
  Preprocessing …
  Training models …

  MODEL COMPARISON
  ...
  >> Apartment in Downtown
     3 bed  |  2 bath  |  1200 sqft

  Estimated Monthly Rent:  ₹32,540
  Typical range:           ₹29,286  -  ₹35,794
```

---

## Dependencies

Install all required packages with:

```bash
pip install scikit-learn pandas numpy xgboost
```

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation & synthetic data generation |
| `pandas` | Data manipulation and feature framing |
| `scikit-learn` | Model training, preprocessing, and evaluation |
| `xgboost` | Gradient Boosting support (imported transitively) |

> **Python version:** 3.8 or higher recommended.

---

## Project Structure

```
house_rent_predictor.py   # Main script — data generation, training, and prediction agent
README.md                 # Project documentation
```
