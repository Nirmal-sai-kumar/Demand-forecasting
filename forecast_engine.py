from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeasonResolution:
    season_name: str
    season_months: list[int]
    mapping_used: dict[str, list[int]]


def compute_months_ahead(last_date: pd.Timestamp, selected_month: int) -> int:
    if last_date is None or pd.isna(last_date):
        raise ValueError("last_date is required")
    if selected_month < 1 or selected_month > 12:
        raise ValueError("selected_month must be between 1 and 12")

    last_month = int(pd.Timestamp(last_date).month)
    if selected_month > last_month:
        return int(selected_month - last_month)
    return int(12 - last_month + selected_month)


def default_season_mapping() -> dict[str, list[int]]:
    # Default mapping chosen to cover all months (common India-like seasons):
    # - Summer: Mar–May
    # - Rainy: Jun–Sep
    # - Winter: Oct–Feb
    return {
        "summer": [3, 4, 5],
        "rainy": [6, 7, 8, 9],
        "winter": [10, 11, 12, 1, 2],
    }


def resolve_season_months(season_name: str, mapping: dict[str, list[int]] | None = None) -> SeasonResolution:
    name = (season_name or "").strip().lower()
    used = mapping or default_season_mapping()
    if name not in used:
        raise ValueError("Unknown season. Use Winter, Summer, or Rainy.")

    months_raw = list(used[name])
    months: list[int] = []
    seen: set[int] = set()
    for m in months_raw:
        try:
            mi = int(m)
        except Exception:
            continue
        if mi < 1 or mi > 12:
            continue
        if mi in seen:
            continue
        seen.add(mi)
        months.append(mi)
    if not months:
        raise ValueError("Season mapping has no months")

    return SeasonResolution(season_name=name, season_months=months, mapping_used=used)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float("nan")
    except Exception:
        return float("nan")


def forecast_monthly_univariate(
    history: list[float],
    predict_one: Callable[[pd.DataFrame], np.ndarray],
    feature_cols: list[str],
    months_ahead: int,
    *,
    weeks_per_month: int = 4,
) -> tuple[list[float], dict[str, Any]]:
    """Iterative forecast using the legacy univariate feature set.

    This intentionally mirrors the existing /upload forecast behavior:
    - Input `history` is a single numeric series (units_sold over time, sorted by date)
    - Iterates `months_ahead * weeks_per_month` steps, appending each prediction back
    - Aggregates 4 steps per month via mean

    Returns (monthly_predictions, meta).
    """

    if months_ahead < 1:
        raise ValueError("months_ahead must be >= 1")

    hist = [float(x) for x in history if np.isfinite(_safe_float(x))]
    if len(hist) == 0:
        raise ValueError("history is empty")

    min_needed = 30
    if len(hist) < min_needed:
        # Fallback: persistence (last known value)
        last_val = float(hist[-1])
        monthly = [last_val for _ in range(months_ahead)]
        return monthly, {"fallback": "persistence", "history_len": len(hist)}

    future_steps = int(months_ahead * weeks_per_month)
    future: list[float] = []

    for _ in range(future_steps):
        lag_1 = float(hist[-1])
        # Guard: if insufficient history for lag_7/rolls, use persistence style.
        if len(hist) >= 7:
            lag_7 = float(hist[-7])
            roll_mean_7 = float(np.mean(hist[-7:]))
        else:
            lag_7 = lag_1
            roll_mean_7 = lag_1

        if len(hist) >= 30:
            roll_mean_30 = float(np.mean(hist[-30:]))
        else:
            roll_mean_30 = roll_mean_7

        features_df = pd.DataFrame(
            [[lag_1, lag_7, roll_mean_7, roll_mean_30]],
            columns=feature_cols,
        )

        pred = predict_one(features_df)
        val = float(pred[0])
        future.append(val)
        hist.append(val)

    monthly_pred = (
        pd.Series(future)
        .groupby(np.arange(len(future)) // weeks_per_month)
        .mean()
    )

    monthly = [float(x) for x in monthly_pred.values.tolist()]
    return monthly, {"fallback": None, "history_len": len(history)}


def parse_iso_date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def month_name(month: int) -> str:
    dt = datetime(2000, month, 1)
    return dt.strftime("%B")
