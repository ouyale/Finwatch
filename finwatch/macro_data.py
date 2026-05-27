"""
macro_data.py - Live ONS macroeconomic data integration.

Fetches current UK economic indicators from the ONS API and caches
them locally. Used to enrich customer features at scoring time and
to trigger retraining when macro conditions shift significantly.

ONS API docs: https://api.ons.gov.uk/
Free, no API key required.

Usage
-----
    from finwatch.macro_data import get_macro_snapshot, check_macro_drift

    # Get latest ONS values (cached for 24h)
    snapshot = get_macro_snapshot()
    # {'cpi_inflation_rate': 3.4, 'unemployment_rate': 4.2, ...}

    # Check if macro conditions have drifted from training baseline
    alert = check_macro_drift(snapshot, baseline_path="data/external/macro_baseline.json")
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from .constants import MACRO_DRIFT_SIGMA, ONS_BASE_URL, ONS_SERIES

logger = logging.getLogger(__name__)

# Cache file - avoids hammering the ONS API on every request
_CACHE_FILE = Path("data/external/ons_cache.json")
_CACHE_TTL_HOURS = 24


def get_macro_snapshot(use_cache: bool = True) -> dict:
    """
    Fetch latest values for all ONS series defined in constants.ONS_SERIES.

    Returns a flat dict of {feature_name: float_value}.
    Falls back to the cached values if the API is unavailable.

    Parameters
    ----------
    use_cache : bool  If True, serve from cache if fresh (< 24h old).
    """
    if use_cache and _cache_is_fresh():
        logger.info("Serving ONS macro snapshot from cache.")
        return _read_cache()["values"]

    snapshot = {}
    for feature_name, series_id in ONS_SERIES.items():
        try:
            value = _fetch_series_latest(series_id)
            snapshot[feature_name] = value
            logger.info("ONS %s (%s) = %.3f", feature_name, series_id, value)
        except Exception as e:
            logger.warning("Failed to fetch ONS series %s: %s - using None", series_id, e)
            snapshot[feature_name] = None

    _write_cache(snapshot)
    return snapshot


def _fetch_series_latest(series_id: str) -> float:
    """
    Fetch the most recent observation for a single ONS time series.

    ONS API endpoint pattern:
      GET /v1/datasets/{series_id}/timeseries/{series_id}/data
    """
    url = f"{ONS_BASE_URL}/datasets/{series_id}/timeseries/{series_id}/data"
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()
    # ONS returns observations sorted oldest-first - take the last one
    observations = data.get("months") or data.get("quarters") or data.get("years") or []
    if not observations:
        raise ValueError(f"No observations found for series {series_id}")

    latest = observations[-1]
    return float(latest["value"])


def save_macro_baseline(snapshot: dict, path: str = "data/external/macro_baseline.json"):
    """
    Save current macro snapshot as the training-time baseline.

    Call this immediately after training a new model so the monitoring
    layer knows what 'normal' looked like at training time.
    """
    baseline = {
        "saved_at": datetime.utcnow().isoformat(),
        "values": snapshot,
        # Placeholder stds - populate with historical data in production
        "stds": {k: 0.5 for k in snapshot},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Macro baseline saved to %s", path)


def check_macro_drift(
    current_snapshot: dict,
    baseline_path: str = "data/external/macro_baseline.json",
    sigma_threshold: float = MACRO_DRIFT_SIGMA,
) -> dict:
    """
    Compare current ONS values against training-time baseline.

    Returns a dict with:
      - drifted_features: list of features that have shifted > sigma_threshold σ
      - max_drift_sigma: the largest drift observed (in standard deviations)
      - should_retrain: bool - True if any feature exceeds the threshold
      - details: per-feature drift values

    Parameters
    ----------
    current_snapshot  : dict from get_macro_snapshot()
    baseline_path     : path to the saved baseline JSON
    sigma_threshold   : number of standard deviations to trigger alert
    """
    if not Path(baseline_path).exists():
        logger.warning("No macro baseline found at %s - skipping drift check.", baseline_path)
        return {"should_retrain": False, "drifted_features": [], "max_drift_sigma": 0.0}

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_values = baseline.get("values", {})
    baseline_stds = baseline.get("stds", {})

    details = {}
    drifted = []

    for feature, current_value in current_snapshot.items():
        if current_value is None:
            continue
        base_val = baseline_values.get(feature)
        base_std = baseline_stds.get(feature, 1.0)

        if base_val is None or base_std == 0:
            continue

        drift_sigma = abs(current_value - base_val) / base_std
        details[feature] = {
            "baseline": base_val,
            "current": current_value,
            "drift_sigma": round(drift_sigma, 3),
            "alert": drift_sigma > sigma_threshold,
        }

        if drift_sigma > sigma_threshold:
            drifted.append(feature)
            logger.warning(
                "Macro drift detected: %s shifted %.2fσ (baseline=%.3f, current=%.3f)",
                feature,
                drift_sigma,
                base_val,
                current_value,
            )

    max_drift = max((d["drift_sigma"] for d in details.values()), default=0.0)

    return {
        "should_retrain": len(drifted) > 0,
        "drifted_features": drifted,
        "max_drift_sigma": round(max_drift, 3),
        "details": details,
        "checked_at": datetime.utcnow().isoformat(),
    }


# -- Cache helpers ------------------------------------------------------------─


def _cache_is_fresh() -> bool:
    if not _CACHE_FILE.exists():
        return False
    cache = _read_cache()
    fetched_at = datetime.fromisoformat(cache.get("fetched_at", "2000-01-01"))
    return datetime.utcnow() - fetched_at < timedelta(hours=_CACHE_TTL_HOURS)


def _read_cache() -> dict:
    with open(_CACHE_FILE) as f:
        return json.load(f)


def _write_cache(values: dict):
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump({"fetched_at": datetime.utcnow().isoformat(), "values": values}, f, indent=2)
