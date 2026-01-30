from flask import Flask, request, render_template, redirect, url_for, session, send_file, after_this_request
import json
import logging
import os
import tempfile
import time
from typing import List, Union
from uuid import uuid4

import pandas as pd
import numpy as np
import joblib

# ---------- RUNTIME DIR (set early for Matplotlib) ----------
RUNTIME_DIR = os.getenv("RUNTIME_DIR", tempfile.gettempdir())
MPLCONFIG_DIR = os.path.join(RUNTIME_DIR, "matplotlib")
os.makedirs(MPLCONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DIR)

# Ensure matplotlib uses a non-GUI backend in production (Render/Linux)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

from supabase_client import get_supabase

# Load secrets/config from .env (do not commit .env)
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)

# ---------- RUNTIME FOLDERS (writable on Render) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(RUNTIME_DIR, "retail_forecast_uploads")
GRAPH_DIR = os.path.join(RUNTIME_DIR, "retail_forecast_graphs")
DOWNLOAD_DIR = os.path.join(RUNTIME_DIR, "retail_forecast_downloads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

ALLOWED_UPLOAD_EXTS = {".csv", ".xlsx", ".xls"}
TREND_EPS_DEFAULT = float(os.getenv("TREND_EPS", "0.01"))
CLEANUP_MAX_AGE_HOURS = float(os.getenv("CLEANUP_MAX_AGE_HOURS", "24"))
MAX_FORECAST_MONTHS = 12

app = Flask(__name__)
app.static_folder = GRAPH_DIR
app.secret_key = os.getenv("FLASK_SECRET_KEY", "retail_secret_key")

# Session cookie defaults (helps login sessions behave consistently)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# Base URL for auth redirects (set in .env if needed)
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/")

# ---------- LOAD MODEL ----------
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model", "xgb_model.joblib"))
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Missing model file: {MODEL_PATH}. "
        "Commit model/xgb_model.joblib to the deployed branch, or set MODEL_PATH."
    )
model = joblib.load(MODEL_PATH)


# XGBoost can raise "data did not contain feature names" on some deployments
# depending on version + pandas integration. Use a safe wrapper that disables
# feature validation when supported.
FEATURE_COLS = ["lag_1", "lag_7", "roll_mean_7", "roll_mean_30"]


def _resp_get(obj, key: str):
    """Safely read a key from supabase-py responses across versions."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first_non_empty(*values: str | None) -> str | None:
    for v in values:
        if v is None:
            continue
        v = str(v).strip()
        if v:
            return v
    return None


def _safe_predict(m, X):
    """Predict with XGBoost models across versions.

    Some XGBoost versions enforce feature-name validation even when given a
    pandas DataFrame. Disabling validate_features avoids Runtime 500s on Render.
    """
    try:
        return m.predict(X, validate_features=False)
    except TypeError:
        return m.predict(X)


def _load_report_from_session() -> dict:
    report_id = session.get("report_id")
    if not report_id:
        raise RuntimeError("No report found in session. Upload a dataset first.")

    report_path = os.path.join(DOWNLOAD_DIR, f"{report_id}.json")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_runtime_dirs(max_age_hours: float = CLEANUP_MAX_AGE_HOURS) -> None:
    """Best-effort cleanup of temp files to prevent disk growth on Render."""
    try:
        max_age_seconds = float(max_age_hours) * 3600.0
    except Exception:
        max_age_seconds = 24.0 * 3600.0

    now = time.time()
    for directory in (UPLOAD_DIR, GRAPH_DIR, DOWNLOAD_DIR):
        try:
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                try:
                    if not os.path.isfile(path):
                        continue
                    age = now - os.path.getmtime(path)
                    if age > max_age_seconds:
                        os.remove(path)
                except OSError:
                    continue
        except OSError:
            continue


def _trend_label(series: Union[pd.Series, np.ndarray, List[float]], eps: float = TREND_EPS_DEFAULT) -> str:
    """Return a human-friendly trend label based on the fitted slope."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return "Insufficient Data"

    x = np.arange(values.size, dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    eps = float(eps)
    if slope > eps:
        return "Upward"
    if slope < -eps:
        return "Downward"
    return "Flat"

# ---------- LOGIN ----------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Supabase Auth expects an email for sign-in.
        email_raw = _first_non_empty(request.form.get('username'), request.form.get('email'))
        pwd = request.form.get('password')
        if not email_raw or not pwd:
            return render_template("login.html", error="Please enter email and password")
        email = email_raw.lower()

        try:
            sb = get_supabase()
            res = sb.auth.sign_in_with_password({"email": email, "password": pwd})
        except Exception as e:
            msg = str(e)
            msg_lower = msg.lower()
            if "email not confirmed" in msg_lower or "email_not_confirmed" in msg_lower:
                return render_template(
                    "login.html",
                    error="Email not confirmed. Please confirm from your inbox, then login."
                )
            if "invalid login credentials" in msg_lower or "invalid_grant" in msg_lower:
                return render_template("login.html", error="Invalid email or password")
            return render_template("login.html", error=f"Supabase login error: {msg}")

        user_obj = _resp_get(res, "user")
        session_obj = _resp_get(res, "session")

        if user_obj is None:
            return render_template("login.html", error="Invalid email or password")

        # If Confirm Email is enabled, Supabase can return a user but no session.
        if session_obj is None:
            return render_template(
                "login.html",
                error="Login blocked. Please confirm your email first, then login."
            )

        access_token = _resp_get(session_obj, "access_token")
        refresh_token = _resp_get(session_obj, "refresh_token")
        if not access_token:
            return render_template("login.html", error="Login failed. Please try again.")

        # Avoid stale data by clearing any previous session before setting new values.
        session.clear()
        session['logged_in'] = True
        session['user_id'] = _resp_get(user_obj, "id")
        session['user_email'] = _resp_get(user_obj, "email")
        session['access_token'] = access_token
        session['refresh_token'] = refresh_token

        # Best-effort: fetch profile (requires your RLS policies + access token)
        try:
            sb_user = get_supabase(session['access_token'])
            profile_res = (
                sb_user.table("profiles")
                .select("id, username")
                .eq("id", session['user_id'])
                .maybe_single()
                .execute()
            )
            if getattr(profile_res, "data", None):
                session['username'] = profile_res.data.get('username')
        except Exception:
            # Don't block login if profile fetch fails
            pass

        return redirect(url_for('home'))
    return render_template("login.html")

# ---------- REGISTER ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Supabase Auth expects an email for sign-up.
        email_raw = _first_non_empty(request.form.get('username'), request.form.get('email'))
        if not email_raw:
            return render_template("register.html", error="Please enter your email")
        email = email_raw.lower()
        # Optional username field (if not provided, derive from email prefix)
        username = (request.form.get('profile_username') or "").strip()
        if not username and '@' in email:
            username = email.split('@', 1)[0]

        pwd = request.form.get('password')
        confirm = request.form.get('confirm') or request.form.get('confirm_password')
        if not pwd or not confirm:
            return render_template("register.html", error="Please enter and confirm your password")

        if pwd != confirm:
            return render_template("register.html", error="Passwords do not match")

        try:
            sb = get_supabase()

            # Send username in user metadata so your DB trigger can populate public.profiles.username
            res = sb.auth.sign_up(
                {
                    "email": email,
                    "password": pwd,
                    "options": {
                        "data": {"username": username},
                        # Ensure the confirmation link redirects back to this Flask app
                        "emailRedirectTo": f"{APP_BASE_URL}/auth/callback",
                    },
                }
            )
        except Exception as e:
            msg = str(e)
            msg_lower = msg.lower()
            if "already registered" in msg_lower or "user already registered" in msg_lower:
                return render_template("register.html", error="User already exists. Please login.")
            if "invalid email" in msg_lower:
                return render_template("register.html", error="Please enter a valid email address")
            if "weak_password" in msg_lower or "password" in msg_lower and "weak" in msg_lower:
                return render_template("register.html", error="Password is too weak. Try a stronger one.")
            return render_template("register.html", error=f"Supabase error: {msg}")

        user_obj = _resp_get(res, "user")
        session_obj = _resp_get(res, "session")

        if user_obj is None:
            return render_template("register.html", error="Registration failed")

        # With email confirmation enabled, session may be None: that's still a success.
        # If email confirmation is disabled, Supabase may return a session; we can auto-login.
        access_token = _resp_get(session_obj, "access_token")
        refresh_token = _resp_get(session_obj, "refresh_token")
        if access_token:
            session.clear()
            session['logged_in'] = True
            session['user_id'] = _resp_get(user_obj, "id")
            session['user_email'] = _resp_get(user_obj, "email")
            session['access_token'] = access_token
            session['refresh_token'] = refresh_token
            return redirect(url_for('home'))

        return render_template(
            "register.html",
            success="Registration successful. Check your email to confirm, then login."
        )

    return render_template("register.html")


@app.route('/auth/callback')
def auth_callback():
    """Landing page for Supabase email confirmation redirect.

    Supabase handles the verification itself and then redirects here.
    We just show a friendly message and send the user to login.
    """
    err = request.args.get('error') or request.args.get('error_code')
    err_desc = request.args.get('error_description')
    if err or err_desc:
        msg = err_desc or err
        return render_template("login.html", error=f"Email confirmation failed: {msg}")

    return render_template(
        "login.html",
        error=None,
        info="Email confirmed. You can login now."
    )


# ---------- FORGOT / RESET PASSWORD ----------
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = (request.form.get('email') or "").strip().lower()
        if not email:
            return render_template("forgot_password.html", error="Please enter your email.")

        try:
            sb = get_supabase()
            redirect_to = f"{APP_BASE_URL}/reset-password"

            # supabase-py has had slightly different method signatures across versions
            try:
                sb.auth.reset_password_email(email, {"redirect_to": redirect_to})
            except TypeError:
                sb.auth.reset_password_email(email)
            except AttributeError:
                # older method name
                sb.auth.reset_password_for_email(email, {"redirect_to": redirect_to})

            return render_template(
                "forgot_password.html",
                info="If an account exists for that email, a password reset link has been sent."
            )
        except Exception as e:
            return render_template("forgot_password.html", error=f"Unable to send reset email: {e}")

    return render_template("forgot_password.html")


@app.route('/reset-password', methods=['GET'])
def reset_password():
    # The Supabase recovery link typically includes tokens in the URL fragment (#...)
    # which the server cannot read. The template uses JS to read the fragment and
    # complete the password update via supabase-js.
    supabase_url = os.getenv("VITE_SUPABASE_URL", "")
    supabase_anon_key = os.getenv("VITE_SUPABASE_ANON_KEY", "")
    if not supabase_url or not supabase_anon_key:
        return render_template(
            "reset_password.html",
            error="Supabase is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.",
            supabase_url="",
            supabase_anon_key="",
        )

    return render_template(
        "reset_password.html",
        supabase_url=supabase_url,
        supabase_anon_key=supabase_anon_key,
    )

# ---------- HOME ----------
@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template("index.html")

# ---------- LOGOUT ----------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------- UPLOAD & FORECAST ----------
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'GET':
        return redirect(url_for('home'))

    app.logger.info("/upload POST received")

    try:
        _cleanup_runtime_dirs()

        file = request.files.get('file')
        if not file or not file.filename:
            return render_template("index.html", error="Please choose a dataset file to upload."), 400

        months_raw = (request.form.get('months') or "").strip()
        if not months_raw.isdigit():
            return render_template("index.html", error="Please enter a valid number of months."), 400
        months = int(months_raw)
        if months < 1 or months > MAX_FORECAST_MONTHS:
            return render_template(
                "index.html",
                error=f"Months must be between 1 and {MAX_FORECAST_MONTHS}.",
            ), 400

        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTS:
            return render_template(
                "index.html",
                error=(
                    f"Unsupported file type: {ext or '(no extension)'}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTS))}."
                ),
            ), 400
        report_id = str(uuid4())

        dataset_path = os.path.join(UPLOAD_DIR, f"{report_id}{ext or '.xlsx'}")
        file.save(dataset_path)

        if ext == ".csv":
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_excel(dataset_path)

        df.columns = df.columns.str.lower().str.strip()

        required_cols = {"date", "units_sold"}
        missing = sorted(required_cols - set(df.columns))
        if missing:
            return render_template(
                "index.html",
                error=f"Missing required columns: {', '.join(missing)}. Required: date, units_sold.",
            ), 400

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
        df = df.sort_values('date').dropna(subset=['date', 'units_sold'])

        raw_history = df['units_sold'].astype(float).tolist()
        if len(raw_history) < 30:
            return render_template(
                "index.html",
                error="Not enough history for forecasting. Please upload at least 30 days of data.",
            ), 400

        # ---------- FEATURE ENGINEERING ----------
        df_feat = df.copy()
        df_feat['lag_1'] = df_feat['units_sold'].shift(1)
        df_feat['lag_7'] = df_feat['units_sold'].shift(7)
        df_feat['roll_mean_7'] = df_feat['units_sold'].shift(1).rolling(7).mean()
        df_feat['roll_mean_30'] = df_feat['units_sold'].shift(1).rolling(30).mean()
        df_feat.dropna(inplace=True)

        if df_feat.empty:
            return render_template(
                "index.html",
                error=(
                    "Not enough rows after feature engineering. "
                    "Please upload a dataset with at least 30+ days of sales history."
                ),
            ), 400

        X = df_feat[FEATURE_COLS].astype(float)
        y = df_feat['units_sold'].astype(float)
        preds = _safe_predict(model, X)

        # ---------- METRICS ----------
        mae = int(mean_absolute_error(y, preds))
        rmse = int(np.sqrt(mean_squared_error(y, preds)))
        r2 = round(r2_score(y, preds), 2)

        # ---------- PAST GRAPH ----------
        graph_df = pd.DataFrame({
            'date': df_feat['date'],
            'actual': y,
            'predicted': preds
        }).set_index('date').resample('W').mean()

        past_actual_trend = _trend_label(graph_df['actual'])
        past_pred_trend = _trend_label(graph_df['predicted'])

        past_image = f"past_{report_id}.png"
        future_image = f"future_{report_id}.png"
        past_path = os.path.join(GRAPH_DIR, past_image)
        future_path = os.path.join(GRAPH_DIR, future_image)

        plt.figure(figsize=(11, 5))
        plt.plot(graph_df.index, graph_df['actual'], label="Actual", linewidth=2)
        plt.plot(graph_df.index, graph_df['predicted'], label="Predicted", linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.title("Actual vs Predicted Sales (Weekly)")
        plt.xlabel("Date")
        plt.ylabel("Units Sold")
        plt.savefig(past_path)
        plt.close()

        # ---------- FUTURE FORECAST ----------
        future: List[float] = []
        history = list(raw_history)

        steps = months * 4
        for _ in range(steps):
            lag_1 = history[-1]
            lag_7 = history[-7]
            roll_mean_7 = float(np.mean(history[-7:]))
            roll_mean_30 = float(np.mean(history[-30:]))

            # XGBoost validates feature names when the model was trained with them.
            # Use a DataFrame with the same column names to avoid ValueError on Render.
            features_df = pd.DataFrame(
                [[lag_1, lag_7, roll_mean_7, roll_mean_30]],
                columns=FEATURE_COLS,
            )
            val = float(_safe_predict(model, features_df)[0])
            future.append(val)
            history.append(val)

        monthly_pred = (
            pd.Series(future)
            .groupby(np.arange(len(future)) // 4)
            .mean()
        )

        future_df = pd.DataFrame({
            "Month": range(1, months + 1),
            "Predicted_Sales": monthly_pred.round().astype(int).values,
        })

        future_forecast_table = future_df.to_dict(orient='records')

        future_trend = _trend_label(monthly_pred.values)

        plt.figure(figsize=(10, 5))
        plt.plot(future_df['Month'], future_df['Predicted_Sales'], marker='o')
        plt.title(f"Future Sales Forecast ({months} Month{'s' if months > 1 else ''})")
        plt.xlabel("Month")
        plt.ylabel("Predicted Sales (Units)")
        plt.xticks(future_df['Month'].astype(int).tolist())
        plt.grid(True)
        plt.savefig(future_path)
        plt.close()

        # ---------- INSIGHTS ----------
        avg_demand = int(future_df['Predicted_Sales'].mean())
        recommendation = (
            "High demand expected. Maintain high inventory."
            if avg_demand > 70 else
            "Moderate demand expected. Maintain balanced inventory."
            if avg_demand > 40 else
            "Low demand expected. Avoid overstocking."
        )

        # ---------- OPTIONAL CATEGORY COST TABLE ----------
        category_cost_table = []
        if 'price' in df.columns and 'product_category' in df.columns:
            df_cat = df.copy()
            df_cat['price'] = pd.to_numeric(df_cat['price'], errors='coerce')
            df_cat['year_month'] = df_cat['date'].dt.to_period('M').astype(str)
            df_cat['total_cost'] = df_cat['price'] * df_cat['units_sold']
            df_cat = df_cat.dropna(subset=['year_month', 'product_category', 'units_sold', 'total_cost'])

            category_cost_df = (
                df_cat.groupby(['year_month', 'product_category'])
                .agg(
                    total_units_sold=('units_sold', 'sum'),
                    total_cost=('total_cost', 'sum')
                )
                .reset_index()
            )

            # Show only the most recent N months (where N == selected forecast months)
            month_keys = sorted(category_cost_df['year_month'].unique())
            last_n = month_keys[-months:] if month_keys else []
            if last_n:
                category_cost_df = category_cost_df[category_cost_df['year_month'].isin(last_n)]

            category_cost_df['total_units_sold'] = category_cost_df['total_units_sold'].round().astype(int)
            category_cost_df['total_cost'] = category_cost_df['total_cost'].round().astype(int)
            category_cost_table = category_cost_df.to_dict(orient='records')

        # ---------- STORE FOR DOWNLOAD (avoid huge cookies) ----------
        report = {
            "metrics": {'MAE': mae, 'RMSE': rmse, 'R2': r2},
            "insights": {'Average Demand': avg_demand, 'Recommendation': recommendation},
            "trends": {
                'Past Actual Trend': past_actual_trend,
                'Past Predicted Trend': past_pred_trend,
                'Future Forecast Trend': future_trend,
            },
            "future_forecast_table": future_forecast_table,
            "category_cost_table": category_cost_table,
            "past_image": past_image,
            "future_image": future_image,
        }

        report_path = os.path.join(DOWNLOAD_DIR, f"{report_id}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f)

        session['report_id'] = report_id

        return render_template(
            "result.html",
            past_image=past_image,
            future_image=future_image,
            months=months,
            mae=mae,
            rmse=rmse,
            r2=r2,
            avg_demand=avg_demand,
            recommendation=recommendation,
            past_actual_trend=past_actual_trend,
            past_pred_trend=past_pred_trend,
            future_trend=future_trend,
            future_forecast_table=future_forecast_table,
            category_cost_table=category_cost_table
        )
    except Exception:
        app.logger.exception("Prediction failed in /upload")
        return render_template("index.html", error="Prediction failed. Please check your dataset format."), 500

# ---------- DOWNLOAD EXCEL ----------
@app.route('/download/excel')
def download_excel():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        report = _load_report_from_session()
    except (RuntimeError, FileNotFoundError, json.JSONDecodeError):
        return render_template("index.html", error="Report not found. Please upload a dataset again."), 400

    report_id = session.get("report_id")
    path = os.path.join(DOWNLOAD_DIR, f"retail_forecast_report_{report_id}.xlsx")

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame([report.get('metrics', {})]).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame([report.get('insights', {})]).to_excel(writer, sheet_name="Insights", index=False)
        pd.DataFrame([report.get('trends', {})]).to_excel(writer, sheet_name="Trends", index=False)
        pd.DataFrame(report.get('future_forecast_table', [])).to_excel(
            writer, sheet_name="Future_Forecast", index=False
        )
        pd.DataFrame(report.get('category_cost_table', [])).to_excel(
            writer, sheet_name="Monthwise_Category_Cost", index=False
        )

    @after_this_request
    def _remove_generated_excel(response):
        try:
            os.remove(path)
        except OSError:
            pass
        return response

    return send_file(path, as_attachment=True)

# ---------- DOWNLOAD PDF ----------
@app.route('/download/pdf')
def download_pdf():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        report = _load_report_from_session()
    except (RuntimeError, FileNotFoundError, json.JSONDecodeError):
        return render_template("index.html", error="Report not found. Please upload a dataset again."), 400

    report_id = session.get("report_id")
    path = os.path.join(DOWNLOAD_DIR, f"retail_forecast_report_{report_id}.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Retail Demand Forecasting Report")

    c.setFont("Helvetica", 11)
    top_margin = 50
    bottom_margin = 50
    y = height - 100

    def ensure_space(needed: float) -> None:
        nonlocal y
        if y - needed < bottom_margin:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - top_margin

    for k, v in (report.get('metrics') or {}).items():
        ensure_space(20)
        c.drawString(50, y, f"{k}: {v}")
        y -= 20

    y -= 20
    for k, v in (report.get('insights') or {}).items():
        ensure_space(20)
        c.drawString(50, y, f"{k}: {v}")
        y -= 20

    # Add trend summary if present
    trends = report.get('trends') or {}
    if trends:
        y -= 10
        ensure_space(40)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Trend Summary")
        y -= 20
        c.setFont("Helvetica", 11)
        for k, v in trends.items():
            ensure_space(18)
            c.drawString(50, y, f"{k}: {v}")
            y -= 18

    # Future forecast values (month -> predicted sales)
    future_table = report.get('future_forecast_table') or []
    if future_table:
        y -= 10
        ensure_space(40)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Future Forecast Values")
        y -= 20
        c.setFont("Helvetica", 11)
        for row in future_table:
            ensure_space(16)
            m = row.get('Month')
            v = row.get('Predicted_Sales')
            c.drawString(50, y, f"Month {m}: {v} units")
            y -= 16

    # Category table (recent N months)
    cat_rows = report.get('category_cost_table') or []
    if cat_rows:
        y -= 10
        ensure_space(40)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Category Cost & Stock (Recent Months)")
        y -= 20
        c.setFont("Helvetica", 10)
        for row in cat_rows:
            ensure_space(14)
            ym = row.get('year_month') or row.get('month')
            cat = row.get('product_category')
            units = row.get('total_units_sold')
            cost = row.get('total_cost')
            c.drawString(50, y, f"{ym} | {cat} | Units: {units} | Cost: {cost}")
            y -= 14

    past_image = report.get("past_image")
    future_image = report.get("future_image")

    image_w = 500
    image_h = 200
    gap = 20

    if past_image:
        ensure_space(image_h + gap + 14)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Past Sales: Actual vs Predicted")
        y -= 14
        c.drawImage(os.path.join(GRAPH_DIR, past_image), 50, y - image_h, image_w, image_h)
        y -= image_h + gap

    if future_image:
        ensure_space(image_h + gap + 14)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Future Forecast")
        y -= 14
        c.drawImage(os.path.join(GRAPH_DIR, future_image), 50, y - image_h, image_w, image_h)
        y -= image_h + gap

    c.save()

    @after_this_request
    def _remove_generated_pdf(response):
        try:
            os.remove(path)
        except OSError:
            pass
        return response

    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
