import os
from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions


def get_supabase(access_token: str | None = None) -> Client:
    url = os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("VITE_SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError(
            "Missing Supabase config. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env"
        )

    # If you pass a user session access token, Supabase will apply RLS policies
    # using auth.uid() for table operations (e.g., public.profiles).
    if access_token:
        options = ClientOptions(headers={"Authorization": f"Bearer {access_token}"})
        return create_client(url, key, options=options)

    return create_client(url, key)
