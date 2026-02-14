import os
from urllib.parse import urlparse

from supabase import Client, create_client


def _read_supabase_config() -> tuple[str, str]:
    # Support common naming conventions.
    url = os.getenv("VITE_SUPABASE_URL") or os.getenv("SUPABASE_URL") or ""
    key = os.getenv("VITE_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY") or ""
    return url, key


_SUPABASE_URL, _SUPABASE_KEY = _read_supabase_config()


def get_supabase(access_token: str | None = None) -> Client:
    url = _SUPABASE_URL
    key = _SUPABASE_KEY

    if not url or not key:
        raise RuntimeError(
            "Missing Supabase config. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY (or SUPABASE_URL/SUPABASE_ANON_KEY) in .env"
        )

    # Catch common copy/paste placeholders early.
    if "YOUR_PROJECT_REF" in url or "YOUR_SUPABASE_ANON_KEY" in key:
        raise RuntimeError(
            "Supabase config looks like placeholders. Replace VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env with real values from your Supabase project settings."
        )

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(
            "Invalid VITE_SUPABASE_URL. It should look like https://<project-ref>.supabase.co"
        )

    sb = create_client(url, key)

    # If you pass a user session access token, apply it to PostgREST so table
    # operations can use RLS policies (auth.uid()).
    if access_token:
        try:
            sb.postgrest.auth(access_token)
        except Exception:
            # Best-effort; callers may still use auth endpoints without PostgREST.
            pass

    return sb
