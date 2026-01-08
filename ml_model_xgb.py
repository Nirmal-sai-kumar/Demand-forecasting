import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = "dataset/retail_sales_forecasting.xlsx"
df = pd.read_excel(file_path)

# -----------------------------
# 2. Basic Cleaning
# -----------------------------
df['date'] = pd.to_datetime(df['date'])   # convert date column
df = df.sort_values('date')               # sort by date
df = df.dropna(subset=['units_sold'])     # remove missing sales rows

# -----------------------------
# 3. Feature Engineering
# -----------------------------

# Lag Features
df['lag_1'] = df['units_sold'].shift(1)
df['lag_7'] = df['units_sold'].shift(7)

# Rolling Features
df['roll_mean_7'] = df['units_sold'].shift(1).rolling(window=7).mean()
df['roll_mean_30'] = df['units_sold'].shift(1).rolling(window=30).mean()

df = df.dropna()  # remove first few rows without lag values

# -----------------------------
# 4. Prepare Data
# -----------------------------
feature_cols = ['lag_1', 'lag_7', 'roll_mean_7', 'roll_mean_30']
X = df[feature_cols]
y = df['units_sold']

# -----------------------------
# 5. Time-Series Validation (Walk-forward)
# -----------------------------
base_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

tscv = TimeSeriesSplit(n_splits=5)
fold_mae = []
fold_rmse = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBRegressor(**base_model.get_params())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    fold_mae.append(mae)
    fold_rmse.append(rmse)

    print(f"Fold {fold_idx}: MAE={mae:.3f} RMSE={rmse:.3f}")

print("\nTime-series CV Summary:")
print(f"MAE:  mean={np.mean(fold_mae):.3f}  std={np.std(fold_mae):.3f}")
print(f"RMSE: mean={np.mean(fold_rmse):.3f} std={np.std(fold_rmse):.3f}")

# -----------------------------
# 6. Train Final Model (all history)
# -----------------------------
final_model = XGBRegressor(**base_model.get_params())
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
if not os.path.exists("model"):
    os.makedirs("model")

joblib.dump(final_model, "model/xgb_model.joblib")

print("\nModel and graph saved successfully.")
