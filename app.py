from flask import Flask, request, render_template, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from supabase_client import get_supabase

# Load secrets/config from .env (do not commit .env)
load_dotenv(override=True)

# ---------- FOLDERS ----------
os.makedirs("downloads", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

app = Flask(__name__)
app.static_folder = 'graphs'
app.secret_key = "retail_secret_key"

# Base URL for auth redirects (set in .env if needed)
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/")

# ---------- LOAD MODEL ----------
model = joblib.load("model/xgb_model.joblib")


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
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    months = int(request.form['months'])

    dataset_path = "dataset/uploaded_data.xlsx"
    file.save(dataset_path)

    df = pd.read_excel(dataset_path)
    df.columns = df.columns.str.lower().str.strip()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').dropna(subset=['units_sold'])

    # ---------- FEATURE ENGINEERING ----------
    df['lag_1'] = df['units_sold'].shift(1)
    df['lag_7'] = df['units_sold'].shift(7)
    df['roll_mean_7'] = df['units_sold'].shift(1).rolling(7).mean()
    df['roll_mean_30'] = df['units_sold'].shift(1).rolling(30).mean()
    df.dropna(inplace=True)

    X = df[['lag_1', 'lag_7', 'roll_mean_7', 'roll_mean_30']]
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

    plt.figure(figsize=(11, 5))
    plt.plot(graph_df.index, graph_df['actual'], label="Actual", linewidth=2)
    plt.plot(graph_df.index, graph_df['predicted'], label="Predicted", linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.title("Actual vs Predicted Sales (Weekly)")
    plt.savefig("graphs/past_graph.png")
    plt.close()

    # ---------- FUTURE FORECAST ----------
    future = []
    history = df['units_sold'].astype(float).tolist()

    # Forecast weekly points (approx. 4 weeks per month)
    steps = months * 4
    for _ in range(steps):
        # Ensure enough history for lag_7 and rolling_30
        # (should already be true due to dropna with rolling_30)
        lag_1 = history[-1]
        lag_7 = history[-7]
        roll_mean_7 = float(np.mean(history[-7:]))
        roll_mean_30 = float(np.mean(history[-30:]))

        features = np.array([[lag_1, lag_7, roll_mean_7, roll_mean_30]], dtype=float)
        val = float(model.predict(features)[0])
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
    plt.savefig("graphs/future_graph.png")
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

    # ==========================================================
    # ✅ REVIEWER FEATURE: MONTH-WISE CATEGORY COST + UNITS SOLD
    # ==========================================================
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

    # ---------- STORE FOR DOWNLOAD ----------
    session['metrics'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    session['insights'] = {'Average Demand': avg_demand, 'Recommendation': recommendation}
    session['trends'] = {
        'Past Actual Trend': past_actual_trend,
        'Past Predicted Trend': past_pred_trend,
        'Future Forecast Trend': future_trend,
    }
    session['category_cost_table'] = category_cost_table

    return render_template(
        "result.html",
        past_image="past_graph.png",
        future_image="future_graph.png",
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

# ---------- DOWNLOAD EXCEL ----------
@app.route('/download/excel')
def download_excel():
    path = "downloads/retail_forecast_report.xlsx"

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame([session['metrics']]).to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame([session['insights']]).to_excel(writer, sheet_name="Insights", index=False)
        pd.DataFrame(session['category_cost_table']).to_excel(
            writer, sheet_name="Monthwise_Category_Cost", index=False
        )

    return send_file(path, as_attachment=True)

# ---------- DOWNLOAD PDF ----------
@app.route('/download/pdf')
def download_pdf():
    path = "downloads/retail_forecast_report.pdf"
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Retail Demand Forecasting Report")

    c.setFont("Helvetica", 11)
    y = height - 100

    for k, v in session['metrics'].items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20

    y -= 20
    for k, v in session['insights'].items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20

    # Add trend summary if present
    trends = session.get('trends') or {}
    if trends:
        y -= 10
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Trend Summary")
        y -= 20
        c.setFont("Helvetica", 11)
        for k, v in trends.items():
            c.drawString(50, y, f"{k}: {v}")
            y -= 18

    c.drawImage("graphs/past_graph.png", 50, 250, 500, 200)
    c.drawImage("graphs/future_graph.png", 50, 20, 500, 200)

    c.save()
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
