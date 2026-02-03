import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""Train XGBoost demand model.

This script supports two modes:

1) Multivariate (recommended): uses store/product + price/promo + calendar features
    if the dataset contains the required columns.
    Output model: model/xgb_model_multifeature.joblib

2) Legacy univariate: uses only lag/rolling features from units_sold.
    Output model: model/xgb_model.joblib

Note: The Flask app defaults to model/xgb_model.joblib. To use the multivariate
model, set MODEL_PATH to the multifeature joblib and update app feature
engineering accordingly.
"""


# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = "dataset/retail_sales_forecasting.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# 2. Basic Cleaning
# -----------------------------
if 'date' not in df.columns or 'units_sold' not in df.columns:
    raise ValueError("Dataset must contain at least: date, units_sold")

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
df = df.dropna(subset=['date', 'units_sold']).sort_values('date')

# -----------------------------
# 3. Feature Engineering
# -----------------------------

def _first_existing(*names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


store_col = _first_existing('store_id', 'store', 'store_code', 'store_name')
product_col = _first_existing('product_category', 'category', 'product_id', 'product')
price_col = _first_existing('price', 'selling_price', 'unit_price')
promo_col = _first_existing('promo', 'promotion', 'is_promo', 'discount_flag')

has_multivariate = all([store_col, product_col, price_col, promo_col])

if has_multivariate:
    # Normalize/clean multivariate columns
    df[store_col] = df[store_col].astype(str).fillna('unknown')
    df[product_col] = df[product_col].astype(str).fillna('unknown')
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    # promo may be 0/1, True/False, Yes/No
    if df[promo_col].dtype == bool:
        df[promo_col] = df[promo_col].astype(int)
    else:
        df[promo_col] = (
            df[promo_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({'true': '1', 'false': '0', 'yes': '1', 'no': '0'})
        )
        df[promo_col] = pd.to_numeric(df[promo_col], errors='coerce')

    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Lag/rolling features per store+product (prevents mixing across categories/stores)
    grp = df.groupby([store_col, product_col], sort=False)
    df['lag_1'] = grp['units_sold'].shift(1)
    df['lag_7'] = grp['units_sold'].shift(7)
    df['roll_mean_7'] = grp['units_sold'].transform(lambda s: s.shift(1).rolling(window=7).mean())
    df['roll_mean_30'] = grp['units_sold'].transform(lambda s: s.shift(1).rolling(window=30).mean())

    df = df.dropna(subset=['lag_1', 'lag_7', 'roll_mean_7', 'roll_mean_30', price_col, promo_col, 'dow', 'month'])
else:
    # Legacy (single-series)
    df['lag_1'] = df['units_sold'].shift(1)
    df['lag_7'] = df['units_sold'].shift(7)
    df['roll_mean_7'] = df['units_sold'].shift(1).rolling(window=7).mean()
    df['roll_mean_30'] = df['units_sold'].shift(1).rolling(window=30).mean()
    df = df.dropna(subset=['lag_1', 'lag_7', 'roll_mean_7', 'roll_mean_30'])

# -----------------------------
# 4. Prepare Data
# -----------------------------
y = df['units_sold'].astype(float)

base_model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

if has_multivariate:
    categorical_features = [store_col, product_col]
    numeric_features = ['lag_1', 'lag_7', 'roll_mean_7', 'roll_mean_30', price_col, promo_col, 'dow', 'month']

    pre = ColumnTransformer(
        transformers=[
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore'),
                categorical_features,
            ),
            (
                'num',
                Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]),
                numeric_features,
            ),
        ],
        remainder='drop',
    )

    model_pipeline = Pipeline(steps=[('pre', pre), ('model', base_model)])
    X = df[categorical_features + numeric_features]
else:
    feature_cols = ['lag_1', 'lag_7', 'roll_mean_7', 'roll_mean_30']
    X = df[feature_cols]
    model_pipeline = base_model

# -----------------------------
# 5. Time-Series Validation (Walk-forward)
# -----------------------------


def _wape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error.

    More stable than MAPE when y_true contains zeros / near-zeros.
    WAPE = sum(|y - yhat|) / sum(|y|) * 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 1e-12:
        return float('nan')
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


tscv = TimeSeriesSplit(n_splits=5)
fold_mae: list[float] = []
fold_rmse: list[float] = []
fold_r2: list[float] = []
fold_wape: list[float] = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Recreate a fresh model each fold
    if has_multivariate:
        model = Pipeline(steps=[('pre', model_pipeline.named_steps['pre']), ('model', XGBRegressor(**base_model.get_params()))])
    else:
        model = XGBRegressor(**base_model.get_params())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))
    wape = _wape_percent(y_val.to_numpy(), np.asarray(y_pred))
    fold_mae.append(float(mae))
    fold_rmse.append(rmse)
    fold_r2.append(r2)
    fold_wape.append(wape)
    print(f"Fold {fold_idx}: MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} WAPE={wape:.2f}%")

print("\nTime-series CV Summary:")
print(f"MAE:  mean={np.mean(fold_mae):.3f}  std={np.std(fold_mae):.3f}")
print(f"RMSE: mean={np.mean(fold_rmse):.3f} std={np.std(fold_rmse):.3f}")
print(f"R2:   mean={np.mean(fold_r2):.3f}  std={np.std(fold_r2):.3f}")
print(f"WAPE: mean={np.mean(fold_wape):.2f}% std={np.std(fold_wape):.2f}%")

# -----------------------------
# 6. Train Final Model (all history)
# -----------------------------
final_model = model_pipeline
final_model.fit(X, y)

# -----------------------------
# 7. Plot In-sample Fit (sanity plot)
# -----------------------------
y_fit = final_model.predict(X)

plt.figure(figsize=(10, 5))
plt.plot(y.values, label='Actual Sales')
plt.plot(y_fit, label='Predicted Sales')
plt.legend()
plt.title("Actual vs Predicted Sales (In-sample)")

# Ensure graphs folder exists
if not os.path.exists("graphs"):
    os.makedirs("graphs")

plt.savefig("graphs/xgb_graph.png")
plt.close()

# -----------------------------
# 8. Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)

model_out = "model/xgb_model_multifeature.joblib" if has_multivariate else "model/xgb_model.joblib"
joblib.dump(final_model, model_out)
print(f"\nModel saved successfully: {model_out}")
print("Graph saved successfully: graphs/xgb_graph.png")
