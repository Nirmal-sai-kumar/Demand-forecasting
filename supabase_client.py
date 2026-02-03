import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions


def get_supabase(access_token: str | None = None) -> Client:
    # Ensure local development can read variables from .env.
    # In production, environment variables should be set by the hosting platform.
    load_dotenv(override=True)

    # Support common naming conventions.
    url = os.getenv("VITE_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.getenv("VITE_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_ANON_KEY")

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

    # If you pass a user session access token, Supabase will apply RLS policies
    # using auth.uid() for table operations (e.g., public.profiles).
    if access_token:
        options = ClientOptions(headers={"Authorization": f"Bearer {access_token}"})
        return create_client(url, key, options=options)

    return create_client(url, key)
