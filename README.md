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

# Required for session + CSRF security (generate a long random value)
FLASK_SECRET_KEY=<generate-a-long-random-secret>

# Required for login/register flows
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_ANON_KEY=<your-supabase-anon-key>

# Optional: Enable emailing the generated PDF report (result page “Email” button)
# Uses SendGrid API (no SMTP needed)
SENDGRID_API_KEY=
SENDGRID_FROM=
```

`VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY` are also accepted as aliases.

#### Generate a strong `FLASK_SECRET_KEY`

Generate a long random secret (recommended) and paste it into `.env`:

Windows PowerShell:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

macOS/Linux:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(64))"
```

Notes:

- Keep it secret. Do not commit `.env` to GitHub.
- If you change `FLASK_SECRET_KEY` later, existing user sessions will be invalidated (users get logged out).
- Windows gotcha: some editors save `.env` as **UTF-8 with BOM**, which can break environment variable names. If your app says the key is missing even though it’s in `.env`, re-save `.env` as **UTF-8 (no BOM)** in VS Code (Encoding → “Save with Encoding…” → UTF-8).

### 4) Ensure the model exists (train if needed)

The app loads the model from `model/xgb_model.json` by default. If that file is missing, train it:

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

- `model/xgb_model_multifeature.json`

Otherwise it trains a legacy univariate model and saves:

- `model/xgb_model.json`


## Supabase Auth Notes (Login/Register/Forgot Password)

This app uses **Supabase Auth** for login/registration.

- Ensure your `.env` contains (server-side names preferred; `VITE_*` names are accepted as aliases):
    - `SUPABASE_URL` (or `VITE_SUPABASE_URL`)
    - `SUPABASE_ANON_KEY` (or `VITE_SUPABASE_ANON_KEY`)
    - `APP_BASE_URL` (your deployed base URL, e.g. `https://your-app.onrender.com`)
    - `FLASK_SECRET_KEY` (recommended for production session security)

On Render, you can set `APP_BASE_URL` explicitly, or rely on Render's built-in `RENDER_EXTERNAL_URL` environment variable (the app will use it automatically if `APP_BASE_URL` is not set).

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
- `FLASK_SECRET_KEY`: Flask session secret (required). Use a long random value and set it as an environment variable in production.
- `MODEL_PATH`: Optional override for the model path (defaults to `model/xgb_model.json`)
- `RUNTIME_DIR`: Optional writable temp directory for graphs/downloads (defaults to OS temp)
- `TREND_EPS`: Optional float threshold for trend labeling (default `0.01`)
- `CLEANUP_MAX_AGE_HOURS`: Optional float age threshold for runtime cleanup (default `24`)

### Render deployment: where to set secrets

On Render, set environment variables in your service:

- Render dashboard → your service → **Environment** → **Add Environment Variable**
- Add `FLASK_SECRET_KEY` (and your Supabase/email/Twilio variables as needed)

Do not rely on a local `.env` file for production deployments.

Emailing the report PDFs (optional):

- `SENDGRID_API_KEY`: SendGrid API key (required to email PDFs)
- `SENDGRID_FROM`: Verified sender email address in SendGrid (required)

## WhatsApp (Twilio) Report Sending (Optional)

The results page includes a **“Send to WhatsApp”** button that sends two generated PDFs (**Report PDF** + **Dashboard PDF**) using the **Twilio WhatsApp Sandbox**.

### Required environment variables

```env
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=

# Sender number (Twilio Sandbox by default)
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886

# MUST be a public HTTPS URL (Twilio must fetch media URLs over HTTPS)
PUBLIC_HTTPS_BASE_URL=https://<public-https-domain>
```

### Local development: get a public HTTPS URL

Twilio cannot download files from `http://127.0.0.1:5000`, so you must use a tunnel.

Option A (no install): **localtunnel**

```powershell
npx --yes localtunnel --port 5000
```

It prints a URL like `https://xxxx.loca.lt`. Set `PUBLIC_HTTPS_BASE_URL` to that value and restart the Flask app.

Option B: **ngrok**

```powershell
ngrok http 5000
```

Set `PUBLIC_HTTPS_BASE_URL` to the `https://...` forwarding URL.

### Sandbox join requirement

The destination WhatsApp number must join your Twilio WhatsApp Sandbox (Twilio Console provides the join code). If not joined, Twilio will reject the send.

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
│   ├── xgb_model.json
│   └── xgb_model_multifeature.json
└── templates/
    ├── index.html
    ├── login.html
    ├── register.html
    ├── forgot_password.html
    ├── reset_password.html
    ├── result.html
    └── upload.html
```
