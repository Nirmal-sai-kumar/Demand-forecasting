from flask import Flask, request, render_template, redirect, url_for, session, send_file
import json
import logging
import os
import tempfile
from uuid import uuid4

import pandas as pd
import numpy as np
import joblib

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
RUNTIME_DIR = os.getenv("RUNTIME_DIR", tempfile.gettempdir())

UPLOAD_DIR = os.path.join(RUNTIME_DIR, "retail_forecast_uploads")
GRAPH_DIR = os.path.join(RUNTIME_DIR, "retail_forecast_graphs")
DOWNLOAD_DIR = os.path.join(RUNTIME_DIR, "retail_forecast_downloads")
MPLCONFIG_DIR = os.path.join(RUNTIME_DIR, "matplotlib")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(MPLCONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DIR)

app = Flask(__name__)
app.static_folder = GRAPH_DIR
app.secret_key = "retail_secret_key"

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


def _load_report_from_session() -> dict:
    report_id = session.get("report_id")
    if not report_id:
        raise RuntimeError("No report found in session. Upload a dataset first.")

    report_path = os.path.join(DOWNLOAD_DIR, f"{report_id}.json")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _trend_label(series: pd.Series | np.ndarray | list[float], eps: float = 1e-9) -> str:
    """Return a human-friendly trend label based on the fitted slope."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return "Flat"

    x = np.arange(values.size, dtype=float)
    slope = np.polyfit(x, values, 1)[0]
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
        email = request.form['username'].strip().lower()
        pwd = request.form['password']

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
            return render_template("login.html", error=f"Supabase login error: {msg}")

        if getattr(res, "user", None) is None:
            return render_template("login.html", error="Invalid credentials")

        # If Confirm Email is enabled, Supabase can return a user but no session.
        if getattr(res, "session", None) is None:
            return render_template(
                "login.html",
                error="Login blocked. Please confirm your email first, then login."
            )

        session['logged_in'] = True
        session['user_id'] = res.user.id
        session['user_email'] = res.user.email
        session['access_token'] = res.session.access_token
        session['refresh_token'] = res.session.refresh_token

        # Best-effort: fetch profile (requires your RLS policies + access token)
        try:
            sb_user = get_supabase(session['access_token'])
            profile_res = (
                sb_user.table("profiles")
                .select("id, username")
                .eq("id", res.user.id)
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
        email = request.form['username'].strip().lower()
        # Optional username field (if not provided, derive from email prefix)
        username = (request.form.get('profile_username') or "").strip()
        if not username and '@' in email:
            username = email.split('@', 1)[0]

        pwd = request.form['password']
        confirm = request.form['confirm']

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
            return render_template("register.html", error=f"Supabase error: {e}")

        if getattr(res, "user", None) is None:
            return render_template("register.html", error="Registration failed")

        # With email confirmation enabled, res.session may be None: that's still a success.
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
        file = request.files.get('file')
        if not file or not file.filename:
            return render_template("index.html", error="Please choose a dataset file to upload."), 400

        months_raw = (request.form.get('months') or "").strip()
        if not months_raw.isdigit():
            return render_template("index.html", error="Please enter a valid number of months."), 400
        months = int(months_raw)
        if months < 1 or months > 24:
            return render_template("index.html", error="Months must be between 1 and 24."), 400

        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
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
        df = df.sort_values('date').dropna(subset=['date', 'units_sold'])

        # ---------- FEATURE ENGINEERING ----------
        df['lag_1'] = df['units_sold'].shift(1)
        df['lag_7'] = df['units_sold'].shift(7)
        df['roll_mean_7'] = df['units_sold'].shift(1).rolling(7).mean()
        df['roll_mean_30'] = df['units_sold'].shift(1).rolling(30).mean()
        df.dropna(inplace=True)

        if df.empty:
            return render_template(
                "index.html",
                error=(
                    "Not enough rows after feature engineering. "
                    "Please upload a dataset with at least 30+ days of sales history."
                ),
            ), 400

        feature_cols = ['lag_1', 'lag_7', 'roll_mean_7', 'roll_mean_30']
        X = df[feature_cols].astype(float)
        y = df['units_sold']
        preds = model.predict(X)

        # ---------- METRICS ----------
        mae = int(mean_absolute_error(y, preds))
        rmse = int(np.sqrt(mean_squared_error(y, preds)))
        r2 = round(r2_score(y, preds), 2)

        # ---------- PAST GRAPH ----------
        graph_df = pd.DataFrame({
            'date': df['date'],
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
        plt.savefig(past_path)
        plt.close()

        # ---------- FUTURE FORECAST ----------
        future: list[float] = []
        history = df['units_sold'].astype(float).tolist()
        if len(history) < 30:
            return render_template(
                "index.html",
                error="Not enough history for forecasting. Please upload more data.",
            ), 400

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
                columns=feature_cols,
            )
            val = float(model.predict(features_df)[0])
            future.append(val)
            history.append(val)

        future_df = pd.DataFrame({
            "Month": range(1, months + 1),
            "Predicted_Sales": (
                pd.Series(future)
                .groupby(np.arange(len(future)) // 4)
                .mean()
                .astype(int)
                .values
            )
        })

        future_trend = _trend_label(future_df['Predicted_Sales'])

        plt.figure(figsize=(10, 5))
        plt.plot(future_df['Month'], future_df['Predicted_Sales'], marker='o')
        plt.title(f"Future Sales Forecast ({months} Month{'s' if months > 1 else ''})")
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
            df['month'] = df['date'].dt.month
            df['total_cost'] = df['price'] * df['units_sold']

            category_cost_df = (
                df.groupby(['month', 'product_category'])
                .agg(
                    total_units_sold=('units_sold', 'sum'),
                    total_cost=('total_cost', 'sum')
                )
                .reset_index()
            )

            category_cost_df['total_units_sold'] = category_cost_df['total_units_sold'].astype(int)
            category_cost_df['total_cost'] = category_cost_df['total_cost'].astype(int)
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
            category_cost_table=category_cost_table
        )
    except Exception:
        app.logger.exception("Prediction failed in /upload")
        return render_template("index.html", error="Prediction failed. Please check your dataset format."), 500

# ---------- DOWNLOAD EXCEL ----------
@app.route('/download/excel')
def download_excel():
    report = _load_report_from_session()
    report_id = session.get("report_id")
    path = os.path.join(DOWNLOAD_DIR, f"retail_forecast_report_{report_id}.xlsx")

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame([report.get('metrics', {})]).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame([report.get('insights', {})]).to_excel(writer, sheet_name="Insights", index=False)
        pd.DataFrame(report.get('category_cost_table', [])).to_excel(
            writer, sheet_name="Monthwise_Category_Cost", index=False
        )

    return send_file(path, as_attachment=True)

# ---------- DOWNLOAD PDF ----------
@app.route('/download/pdf')
def download_pdf():
    report = _load_report_from_session()
    report_id = session.get("report_id")
    path = os.path.join(DOWNLOAD_DIR, f"retail_forecast_report_{report_id}.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Retail Demand Forecasting Report")

    c.setFont("Helvetica", 11)
    y = height - 100

    for k, v in (report.get('metrics') or {}).items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20

    y -= 20
    for k, v in (report.get('insights') or {}).items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20

    # Add trend summary if present
    trends = report.get('trends') or {}
    if trends:
        y -= 10
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Trend Summary")
        y -= 20
        c.setFont("Helvetica", 11)
        for k, v in trends.items():
            c.drawString(50, y, f"{k}: {v}")
            y -= 18

    past_image = report.get("past_image")
    future_image = report.get("future_image")
    if past_image:
        c.drawImage(os.path.join(GRAPH_DIR, past_image), 50, 250, 500, 200)
    if future_image:
        c.drawImage(os.path.join(GRAPH_DIR, future_image), 50, 20, 500, 200)

    c.save()
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
