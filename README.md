# Retail Demand Forecasting

## Setup Instructions

Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train the model:

```powershell
python ml_model_xgb.py
```

Run the web app:

```powershell
python app.py
```

## Supabase Auth Notes (Login/Register/Forgot Password)

This app uses **Supabase Auth** for login/registration.

- Ensure your `.env` contains:
    - `VITE_SUPABASE_URL`
    - `VITE_SUPABASE_ANON_KEY`
    - (recommended) `APP_BASE_URL` (your deployed base URL, e.g. `https://your-app.onrender.com`)

If `APP_BASE_URL` is not set, links sent by email (confirm email / password reset) may redirect to `http://127.0.0.1:5000`.

In Supabase Dashboard → **Auth** → **URL Configuration**, add these to **Additional Redirect URLs**:
- `http://127.0.0.1:5000/auth/callback`
- `http://127.0.0.1:5000/reset-password`
- and your deployed equivalents:
    - `https://<your-domain>/auth/callback`
    - `https://<your-domain>/reset-password`

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
│   └── xgb_model.joblib
└── templates/
    ├── index.html
    ├── login.html
    ├── register.html
    ├── result.html
    └── upload.html
```
