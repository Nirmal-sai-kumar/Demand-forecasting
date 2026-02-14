# Retail Demand Forecasting

Flask web app for retail demand forecasting using an XGBoost model. The UI supports uploading a dataset (CSV/Excel) and forecasting 1–12 months ahead. Authentication (login/register/forgot password) is handled via Supabase Auth.

## Prerequisites

- Python 3.9+ recommended
- Windows PowerShell (commands below use PowerShell)
- A Supabase project (required to log in and use the upload/forecast UI)

## Quickstart (Local)

### 1) Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Configure environment variables (.env)

Create a `.env` file in the project root:

```env
# Local base URL used to construct safe Supabase redirect URLs
APP_BASE_URL=http://127.0.0.1:5000

# Recommended for session security
FLASK_SECRET_KEY=change-me

# Required for login/register flows
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_ANON_KEY=<your-supabase-anon-key>

# Optional: Enable emailing the generated PDF report (result page “Email” button)
# Example (Gmail): SMTP_HOST=smtp.gmail.com, SMTP_PORT=587, SMTP_USE_TLS=true
SMTP_HOST=
SMTP_PORT=587
SMTP_FROM=
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_USE_TLS=true
SMTP_USE_SSL=false
```

`VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY` are also accepted as aliases.

### 4) Ensure the model exists (train if needed)

The app loads the model from `model/xgb_model.joblib` by default. If that file is missing, train it:

```powershell
python ml_model_xgb.py
```

### 5) Run the web app

```powershell
python app.py
```

Open: `http://127.0.0.1:5000`

## Using the App (Upload & Forecast)

  - `date`
  - `units_sold`

### Excel notes
- `.xlsx` is read using `openpyxl`.
- `.xls` is read using `xlrd` (v2+ supports **only** the legacy `.xls` format). If you can, prefer exporting to `.xlsx`.

## Model Training Notes (ml_model_xgb.py)

The training script reads `dataset/retail_sales_forecasting.xlsx` and requires at least:

- `date`
- `units_sold`

If the dataset also contains store/product + price/promo columns, the script will train a multivariate pipeline model and save:

- `model/xgb_model_multifeature.joblib`

Otherwise it trains a legacy univariate model and saves:

- `model/xgb_model.joblib`


## Supabase Auth Notes (Login/Register/Forgot Password)

This app uses **Supabase Auth** for login/registration.

- Ensure your `.env` contains (server-side names preferred; `VITE_*` names are accepted as aliases):
    - `SUPABASE_URL` (or `VITE_SUPABASE_URL`)
    - `SUPABASE_ANON_KEY` (or `VITE_SUPABASE_ANON_KEY`)
    - `APP_BASE_URL` (your deployed base URL, e.g. `https://your-app.onrender.com`)
    - `FLASK_SECRET_KEY` (recommended for production session security)

`APP_BASE_URL` is used to construct safe redirect URLs for Supabase Auth flows (email confirmation / reset password).

In Supabase Dashboard → **Auth** → **URL Configuration**, add these to **Additional Redirect URLs**:

- `http://127.0.0.1:5000/auth/callback`
- `http://127.0.0.1:5000/reset-password`
- and your deployed equivalents:
    - `https://<your-domain>/auth/callback`
    - `https://<your-domain>/reset-password`

## Environment Variables

The app reads configuration from environment variables (and `.env` for local development). Defaults are applied safely if values are missing or invalid.

- `SUPABASE_URL` / `VITE_SUPABASE_URL`: Supabase project URL
- `SUPABASE_ANON_KEY` / `VITE_SUPABASE_ANON_KEY`: Supabase anon key
- `APP_BASE_URL`: Base URL used to construct safe redirect URLs for Supabase email flows
- `FLASK_SECRET_KEY`: Flask session secret (recommended for production session security)
- `MODEL_PATH`: Optional override for the model joblib path (defaults to `model/xgb_model.joblib`)
- `RUNTIME_DIR`: Optional writable temp directory for graphs/downloads (defaults to OS temp)
- `TREND_EPS`: Optional float threshold for trend labeling (default `0.01`)
- `CLEANUP_MAX_AGE_HOURS`: Optional float age threshold for runtime cleanup (default `24`)

Emailing the PDF report (optional):

- `SMTP_HOST`: SMTP server host (required to email PDFs)
- `SMTP_PORT`: SMTP server port (default `587`)
- `SMTP_FROM`: From address (required)
- `SMTP_USERNAME`: SMTP username (optional)
- `SMTP_PASSWORD`: SMTP password / app password (optional)
- `SMTP_USE_TLS`: Use STARTTLS (default `true`)
- `SMTP_USE_SSL`: Use implicit SSL (default `false`)

## Troubleshooting

- PowerShell venv activation blocked: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Login/Register not working: verify `SUPABASE_URL` and `SUPABASE_ANON_KEY` are set and not placeholders
- Redirect errors during email confirmation/reset: ensure `APP_BASE_URL` matches the URLs added in Supabase “Additional Redirect URLs”
- Model missing at startup: run `python ml_model_xgb.py` or set `MODEL_PATH`
- Upload errors: confirm your file has `date` and `units_sold` columns and at least 30 days/rows

## Project Structure

```
Retail_Demand_Forecasting/
│
├── app.py
├── ml_model_xgb.py
├── README.md
├── requirements.txt
├── supabase_client.py
├── users.csv
├── __pycache__/
├── dataset/
├── downloads/
├── graphs/
├── model/
│   ├── xgb_model.joblib
│   └── xgb_model_multifeature.joblib
└── templates/
    ├── index.html
    ├── login.html
    ├── register.html
    ├── forgot_password.html
    ├── reset_password.html
    ├── result.html
    └── upload.html
```
