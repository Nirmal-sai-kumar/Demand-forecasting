from flask import Flask, request, render_template, redirect, url_for, session, send_file, after_this_request
import json
import logging
import os
import re
import tempfile
import threading
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
import matplotlib.dates as mdates

from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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
DEFAULT_TREND_EPS = 0.01
DEFAULT_CLEANUP_MAX_AGE_HOURS = 24.0
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


def _get_env_float(name: str, default: float) -> float:
    """Parse a float environment variable safely.

    - Never raises due to invalid user-provided env var values.
    - Avoids parsing at import time by being called from request-time helpers.
    """
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        value = float(str(raw).strip())
        if not np.isfinite(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(str(raw).strip())
        return value
    except Exception:
        return int(default)

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


def _get_identities_count(user_obj) -> int | None:
    """Best-effort read Supabase user.identities count across versions."""
    if user_obj is None:
        return None
    try:
        identities = _resp_get(user_obj, "identities")
        if identities is None and isinstance(user_obj, dict):
            identities = user_obj.get("identities")
        if identities is None:
            return None
        if isinstance(identities, (list, tuple)):
            return len(identities)
        # Some clients might expose identities as an iterable-like
        try:
            return len(list(identities))
        except Exception:
            return None
    except Exception:
        return None


def _first_non_empty(*values: str | None) -> str | None:
    for v in values:
        if v is None:
            continue
        v = str(v).strip()
        if v:
            return v
    return None


_PASSWORD_MIN_LEN = 8
_PASSWORD_UPPER_RE = re.compile(r"[A-Z]")
_PASSWORD_DIGIT_RE = re.compile(r"\d")
_PASSWORD_SPECIAL_RE = re.compile(r"[^A-Za-z0-9]")


def _sanitize_public_error_message(msg: str | None, limit: int = 300) -> str:
    """Best-effort sanitize exception text for displaying to end users."""
    if not msg:
        return ""
    s = str(msg)
    s = s.replace("\r", " ").replace("\n", " ")
    # Strip some common wrapper prefixes from supabase-py
    s = re.sub(r"AuthApiError\([^)]*\):\s*", "", s)
    s = re.sub(r"APIError\([^)]*\):\s*", "", s)
    s = re.sub(r"HTTPError\([^)]*\):\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > limit:
        s = s[:limit].rstrip() + "…"
    return s


def _password_missing_requirements(password: str) -> list[str]:
    missing: list[str] = []
    if len(password) < _PASSWORD_MIN_LEN:
        missing.append("at least 8 characters")
    if _PASSWORD_UPPER_RE.search(password) is None:
        missing.append("one capital letter (A-Z)")
    if _PASSWORD_DIGIT_RE.search(password) is None:
        missing.append("one number (0-9)")
    if _PASSWORD_SPECIAL_RE.search(password) is None:
        missing.append("one special character (e.g., !@#$)")
    return missing


def _password_strength_error(password: str | None) -> str | None:
    """Return a user-facing error string if password is weak, else None.

    Rules:
    - length >= 8
    - at least 1 uppercase letter
    - at least 1 number
    - at least 1 special character (any non-alphanumeric)
    """
    pw = (password or "")
    if not pw:
        return "Please enter a password."

    missing = _password_missing_requirements(pw)
    if not missing:
        return None

    if missing == ["at least 8 characters"]:
        return "Password is weak: must be at least 8 characters."

    return "Password is weak. Missing: " + ", ".join(missing) + "."


def _friendly_register_error(err: Exception) -> str:
    """Map Supabase sign_up errors to consistent user-friendly messages."""
    raw_msg = _sanitize_public_error_message(str(err))
    msg_lower = raw_msg.lower()

    # Duplicate email (wording varies by supabase-py / GoTrue)
    if (
        "already registered" in msg_lower
        or "user already registered" in msg_lower
        or "already exists" in msg_lower
        or "email already" in msg_lower
        or "email" in msg_lower and "in use" in msg_lower
    ):
        return "This email is already registered. Reset password instead."

    # Weak password (wording varies)
    if (
        "weak_password" in msg_lower
        or ("password" in msg_lower and "weak" in msg_lower)
        or ("password" in msg_lower and "strength" in msg_lower)
        or ("password" in msg_lower and "at least" in msg_lower)
        or ("password" in msg_lower and "short" in msg_lower)
    ):
        return "Password is weak: must be at least 8 characters and include 1 capital letter (A-Z), 1 number (0-9), and 1 special character (e.g., !@#$)."

    if "invalid email" in msg_lower:
        return "Please enter a valid email address"

    if "missing supabase config" in msg_lower:
        return "Registration is temporarily unavailable due to a server configuration issue."

    # Fallback: show the sanitized Supabase error so users know why it failed.
    # (This may still be a little technical, but it's clearer than a generic message.)
    return raw_msg or "Registration failed. Please try again later."


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


def _cleanup_runtime_dirs(max_age_hours: float | None = None) -> None:
    """Best-effort cleanup of temp files to prevent disk growth on Render."""
    if max_age_hours is None:
        max_age_hours = _get_env_float("CLEANUP_MAX_AGE_HOURS", DEFAULT_CLEANUP_MAX_AGE_HOURS)
    max_age_seconds = _get_env_float("CLEANUP_MAX_AGE_HOURS", float(max_age_hours)) * 3600.0

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
                    # Cleanup failures should not affect user-facing responses.
                    app.logger.warning("Cleanup: failed removing file: %s", path)
                    continue
        except OSError:
            app.logger.warning("Cleanup: failed listing dir: %s", directory)
            continue


def _trend_label(series: Union[pd.Series, np.ndarray, List[float]], eps: float | None = None) -> str:
    """Return a human-friendly trend label based on the fitted slope."""
    if eps is None:
        eps = _get_env_float("TREND_EPS", DEFAULT_TREND_EPS)
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


# ------------------------------
# PDF Font caching / idempotent registration
# ------------------------------

_PDF_FONT_LOCK = threading.Lock()
_PDF_FONT_NAMES: tuple[str, str] | None = None


def _looks_like_font_bytes(data: bytes) -> bool:
    """Basic validation to avoid caching HTML error pages as fonts."""
    if not data or len(data) < 12:
        return False
    head = data[:4]
    # TrueType: 0x00010000, OpenType: 'OTTO', TTC: 'ttcf'
    return head in (b"\x00\x01\x00\x00", b"OTTO", b"ttcf")


def _atomic_write_bytes(dest_path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + f".{uuid4().hex}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    # Atomic replace prevents corrupted cache state under concurrent writers.
    os.replace(tmp_path, dest_path)


def _font_file_is_valid(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        return head in (b"\x00\x01\x00\x00", b"OTTO", b"ttcf")
    except OSError:
        return False


def _ensure_cached_font(url: str, dest_path: str, timeout_s: int = 20) -> bool:
    """Ensure a font file exists at dest_path.

    Uses atomic rename to avoid corruption under concurrency.
    Performs simple signature validation; registration will further validate.
    """
    try:
        if os.path.exists(dest_path) and _font_file_is_valid(dest_path):
            return True

        import urllib.request

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "RetailDemandForecasting/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        if not _looks_like_font_bytes(data):
            return False
        _atomic_write_bytes(dest_path, data)
        return _font_file_is_valid(dest_path)
    except Exception:
        app.logger.exception("Failed ensuring cached font: %s", url)
        return False


def _register_ttf_if_needed(font_name: str, font_path: str) -> bool:
    """Register a TTF/OTF font idempotently (safe to call multiple times)."""
    try:
        if not font_path or not os.path.exists(font_path):
            return False
        if font_name in pdfmetrics.getRegisteredFontNames():
            return True
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        return True
    except Exception:
        app.logger.exception("Failed registering TTF font: %s", font_path)
        return False


def _get_pdf_font_names() -> tuple[str, str]:
    """Return (regular_font_name, bold_font_name) for PDF rendering.

    - Caches discovery results per-process to avoid repeated filesystem scans.
    - Ensures consistent regular/bold pairing.
    - Safe under concurrent requests.
    """
    global _PDF_FONT_NAMES
    with _PDF_FONT_LOCK:
        if _PDF_FONT_NAMES is not None:
            return _PDF_FONT_NAMES

        font_regular = "Helvetica"
        font_bold = "Helvetica-Bold"

        base_dir = os.path.dirname(os.path.abspath(__file__))

        repo_regular = [
            os.path.join(base_dir, "static", "fonts", "DejaVuSans.ttf"),
            os.path.join(base_dir, "static", "fonts", "NotoSans-Regular.ttf"),
            os.path.join(base_dir, "fonts", "DejaVuSans.ttf"),
            os.path.join(base_dir, "fonts", "NotoSans-Regular.ttf"),
        ]
        repo_bold = [
            os.path.join(base_dir, "static", "fonts", "DejaVuSans-Bold.ttf"),
            os.path.join(base_dir, "static", "fonts", "NotoSans-Bold.ttf"),
            os.path.join(base_dir, "fonts", "DejaVuSans-Bold.ttf"),
            os.path.join(base_dir, "fonts", "NotoSans-Bold.ttf"),
        ]

        linux_regular = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        linux_bold = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]

        windir = os.environ.get("WINDIR", r"C:\\Windows")
        fonts_dir = os.path.join(windir, "Fonts")
        windows_regular = [
            os.path.join(fonts_dir, "segoeui.ttf"),
            os.path.join(fonts_dir, "seguisym.ttf"),
            os.path.join(fonts_dir, "arial.ttf"),
        ]
        windows_bold = [
            os.path.join(fonts_dir, "segoeuib.ttf"),
            os.path.join(fonts_dir, "arialbd.ttf"),
        ]

        regular_candidates = repo_regular + linux_regular + windows_regular
        bold_candidates = repo_bold + linux_bold + windows_bold

        # Prefer system/repo fonts first (no network dependency)
        if any(_register_ttf_if_needed("PDFUnicode", p) for p in regular_candidates):
            font_regular = "PDFUnicode"
        if any(_register_ttf_if_needed("PDFUnicode-Bold", p) for p in bold_candidates):
            font_bold = "PDFUnicode-Bold"

        # If bold isn't available but regular is, keep pairing consistent.
        if font_regular != "Helvetica" and font_bold in ("Helvetica-Bold", "Helvetica"):
            font_bold = font_regular

        # Final fallback: cache-download DejaVu into runtime dir.
        if font_regular == "Helvetica":
            fonts_cache_dir = os.path.join(RUNTIME_DIR, "pdf_fonts")
            dejavu_regular_path = os.path.join(fonts_cache_dir, "DejaVuSans.ttf")
            dejavu_bold_path = os.path.join(fonts_cache_dir, "DejaVuSans-Bold.ttf")
            dejavu_regular_url = "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/version_2_37/ttf/DejaVuSans.ttf"
            dejavu_bold_url = "https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/version_2_37/ttf/DejaVuSans-Bold.ttf"

            if _ensure_cached_font(dejavu_regular_url, dejavu_regular_path) and _register_ttf_if_needed(
                "PDFUnicode", dejavu_regular_path
            ):
                font_regular = "PDFUnicode"
            if _ensure_cached_font(dejavu_bold_url, dejavu_bold_path) and _register_ttf_if_needed(
                "PDFUnicode-Bold", dejavu_bold_path
            ):
                font_bold = "PDFUnicode-Bold"

            if font_regular != "Helvetica" and font_bold in ("Helvetica-Bold", "Helvetica"):
                font_bold = font_regular

        _PDF_FONT_NAMES = (font_regular, font_bold)
        return _PDF_FONT_NAMES

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
            # Avoid leaking internal configuration or stack traces to users.
            app.logger.exception("Supabase sign-in failed")
            msg_lower = str(e).lower()
            if "email not confirmed" in msg_lower or "email_not_confirmed" in msg_lower:
                return render_template("login.html", error="Email not confirmed. Please confirm from your inbox, then login.")
            if "invalid login credentials" in msg_lower or "invalid_grant" in msg_lower:
                return render_template("login.html", error="Invalid email or password")
            if "missing supabase config" in msg_lower:
                return render_template("login.html", error="Login is temporarily unavailable due to a server configuration issue.")
            return render_template("login.html", error="Login failed. Please try again later.")

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
            # Don't block login if profile fetch fails, but don't fail silently.
            app.logger.warning("Profile fetch failed for user_id=%s", session.get('user_id'), exc_info=True)

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

        strength_err = _password_strength_error(pwd)
        if strength_err:
            return render_template("register.html", error=strength_err)

        try:
            sb = get_supabase()

            # Never construct redirect URLs from request headers (open redirect risk).
            redirect_base = APP_BASE_URL

            # Send username in user metadata so your DB trigger can populate public.profiles.username
            res = sb.auth.sign_up(
                {
                    "email": email,
                    "password": pwd,
                    "options": {
                        "data": {"username": username},
                        # Ensure the confirmation link redirects back to this Flask app
                        "emailRedirectTo": f"{redirect_base}/auth/callback",
                    },
                }
            )
        except Exception as e:
            app.logger.exception("Supabase sign-up failed")
            return render_template("register.html", error=_friendly_register_error(e))

        user_obj = _resp_get(res, "user")
        session_obj = _resp_get(res, "session")

        if user_obj is None:
            return render_template("register.html", error="Registration failed")

        # Some Supabase/Auth configurations return 200 with a user but no new identity
        # when the email already exists. Treat that as a duplicate-email outcome so
        # the UI doesn't misleadingly say "Registration successful".
        identities_count = _get_identities_count(user_obj)
        if session_obj is None and identities_count == 0:
            return render_template(
                "register.html",
                error="This email is already registered. Reset password instead.",
            )

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
            # Never construct redirect URLs from request headers (open redirect risk).
            redirect_base = APP_BASE_URL
            redirect_to = f"{redirect_base}/reset-password"

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
            app.logger.exception("Supabase reset password email failed")
            msg_lower = str(e).lower()
            if "missing supabase config" in msg_lower:
                return render_template("forgot_password.html", error="Password reset is temporarily unavailable due to a server configuration issue.")
            return render_template("forgot_password.html", error="Unable to send reset email. Please try again later.")

    return render_template("forgot_password.html")


@app.route('/reset-password', methods=['GET'])
def reset_password():
    # The Supabase recovery link typically includes tokens in the URL fragment (#...)
    # which the server cannot read. The template uses JS to read the fragment and
    # complete the password update via supabase-js.
    # Support both server-side and Vite-style variable names.
    supabase_url = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL") or ""
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY") or ""
    if not supabase_url or not supabase_anon_key:
        app.logger.error("Reset password page requested but Supabase env vars are missing")
        return render_template(
            "reset_password.html",
            error="Password reset is not configured on this server.",
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

        try:
            if ext == ".csv":
                df = pd.read_csv(dataset_path)
            elif ext == ".xls":
                # .xls requires xlrd; keep error message user-friendly.
                df = pd.read_excel(dataset_path, engine="xlrd")
            else:
                df = pd.read_excel(dataset_path)
        except Exception:
            app.logger.exception("Failed reading uploaded file: %s", filename)
            return render_template(
                "index.html",
                error="Unable to read the uploaded file. Please upload a valid .csv or Excel file.",
            ), 400

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
        # Show month names only (no year) on the x-axis
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.xticks(rotation=0)
        plt.tight_layout()
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
            "Month": list(range(1, months + 1)),
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
        plt.tight_layout()
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

        # ---------- OPTIONAL CATEGORY COST TABLE (FUTURE FORECAST MONTHS) ----------
        # Requirement: show forecast months (1..N), NOT recent historical months, and month as numbers only.
        category_cost_table = []
        if 'price' in df.columns and 'product_category' in df.columns:
            df_cat = df[['date', 'product_category', 'units_sold', 'price']].copy()
            df_cat['date'] = pd.to_datetime(df_cat['date'], errors='coerce')
            df_cat['price'] = pd.to_numeric(df_cat['price'], errors='coerce')
            df_cat['units_sold'] = pd.to_numeric(df_cat['units_sold'], errors='coerce')
            df_cat = df_cat.dropna(subset=['date', 'product_category', 'units_sold', 'price'])

            if not df_cat.empty:
                # Build historical monthly totals per category
                df_cat['year_month'] = df_cat['date'].dt.to_period('M')
                df_cat['total_cost'] = df_cat['price'] * df_cat['units_sold']

                monthly_cat = (
                    df_cat.groupby(['year_month', 'product_category'])
                    .agg(
                        units=('units_sold', 'sum'),
                        cost=('total_cost', 'sum'),
                    )
                    .reset_index()
                )

                # Use the most recent months available (up to the requested forecast horizon)
                month_keys = sorted(monthly_cat['year_month'].unique())
                recent_months = month_keys[-months:] if month_keys else []
                if recent_months:
                    recent = monthly_cat[monthly_cat['year_month'].isin(recent_months)].copy()
                else:
                    recent = monthly_cat.copy()

                if not recent.empty:
                    # Average monthly units per category from the uploaded dataset
                    avg_units_by_cat = recent.groupby('product_category')['units'].mean()
                    baseline_total = float(avg_units_by_cat.sum())

                    # Units-weighted average price per category from the uploaded dataset
                    price_weighted = df_cat.groupby('product_category').apply(
                        lambda g: (g['price'] * g['units_sold']).sum() / max(g['units_sold'].sum(), 1.0)
                    )
                    price_weighted = price_weighted.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                    if baseline_total > 0:
                        # Scale using the forecast's *relative* month-to-month changes,
                        # but keep the magnitude based on the uploaded dataset.
                        forecast_vals = future_df['Predicted_Sales'].astype(float).values
                        denom = float(np.mean(forecast_vals)) if forecast_vals.size else 1.0
                        denom = denom if denom != 0 else 1.0

                        categories = avg_units_by_cat.index.tolist()

                        for i in range(1, months + 1):
                            rel = float(forecast_vals[i - 1]) / denom
                            target_total = baseline_total * rel

                            # Allocate to categories proportionally to historical averages
                            remaining = target_total
                            for idx, cat in enumerate(categories):
                                if idx == len(categories) - 1:
                                    units = max(0, int(round(remaining)))
                                else:
                                    share = float(avg_units_by_cat.loc[cat]) / baseline_total
                                    units = max(0, int(round(target_total * share)))
                                    remaining -= units

                                avg_price = float(price_weighted.get(cat, 0.0))
                                total_cost = int(round(units * avg_price))
                                category_cost_table.append(
                                    {
                                        'month': i,
                                        'product_category': str(cat),
                                        'total_units_sold': int(units),
                                        'total_cost': int(total_cost),
                                    }
                                )

        # ---------- STORE FOR DOWNLOAD (avoid huge cookies) ----------
        report = {
            "months": months,
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

    response = send_file(path, as_attachment=True)

    @response.call_on_close
    def _remove_generated_excel() -> None:
        try:
            os.remove(path)
        except OSError:
            # Cleanup failures must not affect user response.
            app.logger.warning("Failed removing generated Excel: %s", path)

    return response

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

    # Prefer a Unicode-capable TrueType font so the Rupee symbol (₹) renders correctly.
    # Cached per-process to avoid repeated filesystem scans and repeated registration.
    FONT_REGULAR, FONT_BOLD = _get_pdf_font_names()

    top_margin = 50
    bottom_margin = 50
    left = 50
    right = 50

    def new_page(font_size: int = 11) -> float:
        c.showPage()
        # Explicitly restore canvas state to avoid relying on ReportLab defaults.
        c.setLineWidth(1)
        c.setFont(FONT_REGULAR, font_size)
        return height - top_margin

    def ensure_space(y: float, needed: float, font_size: int = 11) -> float:
        if y - needed < bottom_margin:
            return new_page(font_size=font_size)
        return y

    def draw_section_title(y: float, title: str) -> float:
        y = ensure_space(y, 24, font_size=11)
        c.setFont(FONT_BOLD, 12)
        c.drawString(left, y, title)
        c.setFont(FONT_REGULAR, 11)
        return y - 18

    def draw_centered_title(y: float, title: str) -> float:
        y = ensure_space(y, 30, font_size=11)
        c.setFont(FONT_BOLD, 16)
        c.drawCentredString(width / 2, y, title)
        c.setFont(FONT_REGULAR, 11)
        return y - 28

    def draw_kv_lines(y: float, items: dict, line_height: float = 16) -> float:
        for k, v in items.items():
            y = ensure_space(y, line_height)
            c.drawString(left, y, f"{k}: {v}")
            y -= line_height
        return y

    def draw_table(
        y: float,
        headers: list[str],
        rows: list[list[str]],
        col_widths: list[float],
        row_h: float = 22,
        grid_line_width: float = 2.2,
        outer_line_width: float = 2.8,
    ) -> float:
        """Draw a paginated table.

        - Repeats header on each page
        - Keeps borders tight to rendered rows (no extra empty boxed area)
        """

        table_w = sum(col_widths)

        def _baseline(y_top: float, h: float, font_size: int) -> float:
            # y_top is the top boundary of the row. drawString uses baseline.
            return y_top - h + (h - font_size) / 2 + 3

        def _draw_header(y_top: float) -> float:
            c.setFont(FONT_BOLD, 11)
            x = left
            for header, w in zip(headers, col_widths):
                c.rect(x, y_top - row_h, w, row_h, stroke=1, fill=0)
                c.drawCentredString(x + (w / 2), _baseline(y_top, row_h, 11), str(header))
                x += w
            return y_top - row_h

        def _draw_row(y_top: float, row: list[str]) -> float:
            c.setFont(FONT_REGULAR, 11)
            x = left
            for cell, w in zip(row, col_widths):
                c.rect(x, y_top - row_h, w, row_h, stroke=1, fill=0)
                c.drawCentredString(x + (w / 2), _baseline(y_top, row_h, 11), str(cell))
                x += w
            return y_top - row_h

        # Always restore to a known line width; avoid accessing ReportLab internals.
        c.setLineWidth(grid_line_width)

        # Ensure we have room for at least header + one row; otherwise start a new page.
        min_needed = row_h * 2
        y = ensure_space(y, min_needed)

        segment_top = y
        segment_rows = 0
        y = _draw_header(y)

        for row in rows:
            # Need space for the next row (and a small buffer) on the current page.
            if y - row_h < bottom_margin:
                # Close current page segment with an outer border matching rendered height.
                segment_h = row_h * (1 + segment_rows)
                c.setLineWidth(outer_line_width)
                c.rect(left, segment_top - segment_h, table_w, segment_h, stroke=1, fill=0)
                c.setLineWidth(grid_line_width)

                # Next page segment
                y = new_page(font_size=11)
                y = ensure_space(y, min_needed)
                segment_top = y
                segment_rows = 0
                y = _draw_header(y)

            y = _draw_row(y, row)
            segment_rows += 1

        # Close final segment border
        segment_h = row_h * (1 + segment_rows)
        c.setLineWidth(outer_line_width)
        c.rect(left, segment_top - segment_h, table_w, segment_h, stroke=1, fill=0)

        c.setLineWidth(1)
        c.setFont(FONT_REGULAR, 11)
        return y - 10

    # ---------- PAGE 1: Title + Graphs first ----------
    c.setFont(FONT_BOLD, 14)
    c.drawString(left, height - 50, "Retail Demand Forecasting Report")

    y = height - 80
    image_w = width - left - right
    image_h = 240
    gap = 18

    past_image = report.get("past_image")
    future_image = report.get("future_image")
    months = int(report.get("months") or 0)

    if past_image:
        y = ensure_space(y, image_h + gap + 18)
        c.setFont(FONT_BOLD, 12)
        c.drawString(left, y, "Actual vs Predicted Sales (Weekly Averages)")
        y -= 16
        c.drawImage(
            os.path.join(GRAPH_DIR, past_image),
            left,
            y - image_h,
            image_w,
            image_h,
            preserveAspectRatio=True,
            anchor='c',
        )
        y -= image_h + gap

    if future_image:
        y = ensure_space(y, image_h + gap + 18)
        title = "Future Sales Forecast"
        if months:
            title = f"Future Sales Forecast ({months} Month{'s' if months > 1 else ''})"
        c.setFont(FONT_BOLD, 12)
        c.drawString(left, y, title)
        y -= 16
        c.drawImage(
            os.path.join(GRAPH_DIR, future_image),
            left,
            y - image_h,
            image_w,
            image_h,
            preserveAspectRatio=True,
            anchor='c',
        )
        y -= image_h + gap

    # Start details after graphs (new page so opening the PDF shows graphs first)
    y = new_page(font_size=11)

    # ---------- BUSINESS INSIGHTS ----------
    insights = report.get('insights') or {}
    if insights:
        y = draw_section_title(y, "Business Insights & Recommendation")
        y = draw_kv_lines(y, insights)
        y -= 8

    # ---------- TREND SUMMARY ----------
    trends = report.get('trends') or {}
    if trends:
        y = draw_section_title(y, "Trend Summary")
        y = draw_kv_lines(y, trends, line_height=15)
        y -= 8

    # ---------- CATEGORY COST & STOCK TABLE ----------
    cat_rows = report.get('category_cost_table') or []
    if cat_rows:
        # Put the report table on a dedicated page, with a header line above the table (same page).
        y = new_page(font_size=11)
        section_title = "Category Cost & Stock Analysis (Forecast Months)"
        y = draw_centered_title(y, section_title)
        headers = ["Month", "Product Category", "Units Sold", "Total Cost (₹)"]
        rows = []
        for r in cat_rows:
            m = r.get('month') or ""
            rows.append([
                str(m),
                str(r.get('product_category', '')),
                str(r.get('total_units_sold', '')),
                f"₹ {r.get('total_cost', '')}",
            ])
        col_widths = [70, width - left - right - (70 + 90 + 120), 90, 120]
        y = draw_table(y, headers, rows, col_widths=col_widths)

    c.save()

    response = send_file(path, as_attachment=True)

    @response.call_on_close
    def _remove_generated_pdf() -> None:
        try:
            os.remove(path)
        except OSError:
            # Cleanup failures must not affect user response.
            app.logger.warning("Failed removing generated PDF: %s", path)

    return response

if __name__ == "__main__":
    app.run(debug=True)
