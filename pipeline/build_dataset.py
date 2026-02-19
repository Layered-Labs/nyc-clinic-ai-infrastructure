#!/usr/bin/env python3
"""
build_dataset.py
================
NYC Clinic AI Infrastructure Dataset

Builds a per-ZIP-code CSV with the infrastructure dimensions needed to
evaluate whether a clinic can run local AI models on-premise:

  1. Internet access  — US Census ACS 5-yr 2022 (public API, no auth required)
  2. Electricity cost — EIA Open Data API (free key) or hardcoded NY avg fallback
  3. Electricity reliability — EIA Form 861 2024 bulk download (no auth)

Output: outputs/nyc_clinic_infrastructure.csv

Setup:
  pip install -r requirements_dataset.txt
  export EIA_API_KEY=your_key     # optional — https://www.eia.gov/opendata/register.php
  export CENSUS_API_KEY=your_key  # optional — https://api.census.gov/data/key_signup.html

Run:
  python build_dataset.py

NOTE — FCC National Broadband Map:
  The FCC Broadband Map API (broadbandmap.fcc.gov) requires a free account for
  bulk data download. To add ISP-level coverage columns, download the NY state
  availability CSV from https://broadbandmap.fcc.gov/data-download and join on
  `zipcode`. Column stubs are included in the output as NaN placeholders.

Data vintage: ACS 2022 5-yr, EIA retail prices 2023, EIA Form 861 2024
"""

import io
import json
import logging
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import pgeocode
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Directories ────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("outputs")
CACHE_DIR  = Path("cache")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── API keys ───────────────────────────────────────────────────────────────────

EIA_API_KEY    = os.environ.get("EIA_API_KEY", "")
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")

# ── Borough ↔ County mapping ───────────────────────────────────────────────────

COUNTY_TO_BOROUGH = {
    "New York": "Manhattan",
    "Bronx":    "Bronx",
    "Kings":    "Brooklyn",
    "Queens":   "Queens",
    "Richmond": "Staten Island",
}
NYC_COUNTIES = set(COUNTY_TO_BOROUGH.keys())

# ── NYC primary utility ────────────────────────────────────────────────────────
# Consolidated Edison serves all 5 boroughs.
# Orange & Rockland serves a small outer fringe — negligible for clinic dataset.

NYC_UTILITY_NAME = "Consolidated Edison Co-NY Inc"
NYC_UTILITY_NUM  = 4226   # EIA Utility Number (Form 861)

# ── Hardcoded EIA fallback (EIA Electric Power Monthly, Oct 2024, NY) ─────────

EIA_FALLBACK = {
    "commercial_rate_cents_kwh":  17.82,
    "residential_rate_cents_kwh": 22.14,
}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — NYC ZIP codes
# ══════════════════════════════════════════════════════════════════════════════

def get_nyc_zipcodes() -> pd.DataFrame:
    """
    Return DataFrame of NYC ZIP codes with centroid lat/lon and borough,
    sourced from the pgeocode offline database (GeoNames / US Census TIGER).
    """
    log.info("Loading NYC ZIP codes …")
    nom = pgeocode.Nominatim("us")
    all_ny = nom._data[nom._data["state_code"] == "NY"]

    nyc = (
        all_ny
        .query("county_name in @NYC_COUNTIES")
        [["postal_code", "place_name", "county_name", "latitude", "longitude"]]
        .rename(columns={
            "postal_code": "zipcode",
            "place_name":  "neighborhood",
            "county_name": "county",
            "latitude":    "lat",
            "longitude":   "lon",
        })
        .dropna(subset=["lat", "lon"])
        .copy()
    )

    nyc["borough"] = nyc["county"].map(COUNTY_TO_BOROUGH)
    nyc             = nyc.reset_index(drop=True)

    log.info(f"  {len(nyc)} ZIP codes — {nyc['borough'].value_counts().to_dict()}")
    return nyc


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — US Census ACS: Internet access
# ══════════════════════════════════════════════════════════════════════════════
# Table B28002: Presence and Types of Internet Subscriptions in Household
# Source: https://www.census.gov/data/developers/data-sets/acs-5year.html
#
# Variables used (labels verified via Census API variables.json):
#   B28002_001E  Total households
#   B28002_002E  With any internet subscription (incl. cellular)
#   B28002_007E  Fixed broadband: cable, fiber optic, or DSL (key metric for clinics)
#   B28002_013E  No Internet access at all
#
# B28002_007E is the most clinically relevant metric: a cellular-only connection
# is insufficient for a clinic's operational needs. Fixed broadband (cable/fiber/DSL)
# is required for reliable local-model updates, EHR sync, and telehealth.
#
# NOTE: Census uses ZCTAs (ZIP Code Tabulation Areas), which approximate but
# do not perfectly match USPS ZIP codes. Most NYC ZIP codes have a matching
# ZCTA. Unmatched ZIPs will show NaN internet columns.

CENSUS_URL  = "https://api.census.gov/data/2022/acs/acs5"
CENSUS_VARS = "B28002_001E,B28002_002E,B28002_007E,B28002_013E"


def get_census_internet(nyc_zips: set) -> pd.DataFrame:
    cache = CACHE_DIR / "census_internet_all_us.json"

    if cache.exists():
        log.info("Census ACS: loading from cache …")
        raw = json.loads(cache.read_text())
    else:
        log.info("Fetching Census ACS internet data for all US ZCTAs …")
        params = {"get": CENSUS_VARS, "for": "zip code tabulation area:*"}
        if CENSUS_API_KEY:
            params["key"] = CENSUS_API_KEY
        try:
            r = requests.get(CENSUS_URL, params=params, timeout=60)
            r.raise_for_status()
            raw = r.json()
            cache.write_text(json.dumps(raw))
            log.info(f"  Downloaded {len(raw)-1} ZCTAs")
        except Exception as e:
            log.error(f"  Census API failed: {e}")
            return pd.DataFrame(columns=[
                "zipcode", "census_hh_total", "census_hh_any_internet",
                "census_hh_broadband", "census_hh_no_internet",
                "census_broadband_pct", "census_no_internet_pct",
            ])

    headers = raw[0]
    df = pd.DataFrame(raw[1:], columns=headers)
    df = df.rename(columns={
        "zip code tabulation area": "zipcode",
        "B28002_001E": "census_hh_total",
        "B28002_002E": "census_hh_any_subscription",
        "B28002_007E": "census_hh_fixed_broadband",   # cable / fiber / DSL
        "B28002_013E": "census_hh_no_internet",
    })

    for col in ["census_hh_total", "census_hh_any_subscription",
                "census_hh_fixed_broadband", "census_hh_no_internet"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)

    df["census_fixed_broadband_pct"] = (
        df["census_hh_fixed_broadband"] / df["census_hh_total"] * 100
    ).round(1)
    df["census_no_internet_pct"] = (
        df["census_hh_no_internet"] / df["census_hh_total"] * 100
    ).round(1)

    # Filter to NYC ZCTAs
    nyc_df = df[df["zipcode"].isin(nyc_zips)].reset_index(drop=True)
    log.info(f"  {len(nyc_df)} ZCTAs matched in NYC (of {len(nyc_zips)} ZIP codes)")

    return nyc_df[[
        "zipcode", "census_hh_total", "census_hh_any_subscription",
        "census_hh_fixed_broadband", "census_hh_no_internet",
        "census_fixed_broadband_pct", "census_no_internet_pct",
    ]]


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — EIA: Electricity retail prices
# ══════════════════════════════════════════════════════════════════════════════
# API docs: https://www.eia.gov/opendata/
# Returns average retail price (¢/kWh) by state × sector × year.

EIA_BASE = "https://api.eia.gov/v2"


def get_eia_rates() -> dict:
    """Return NY commercial and residential electricity rates (¢/kWh)."""
    if not EIA_API_KEY:
        log.warning("EIA_API_KEY not set — using hardcoded Oct 2024 NY averages")
        log.warning("  Get a free key: https://www.eia.gov/opendata/register.php")
        return EIA_FALLBACK.copy()

    cache = CACHE_DIR / "eia_rates_ny.json"
    if cache.exists():
        log.info("EIA rates: loading from cache …")
        return json.loads(cache.read_text())

    log.info("Fetching EIA electricity rates for NY …")
    results = {}

    for sector, key in [
        ("commercial",   "commercial_rate_cents_kwh"),
        ("residential",  "residential_rate_cents_kwh"),
    ]:
        url = f"{EIA_BASE}/electricity/retail-sales/data/"
        params = {
            "api_key":              EIA_API_KEY,
            "frequency":            "annual",
            "data[0]":              "price",
            "facets[stateid][]":    "NY",
            "facets[sectorName][]": sector,
            "sort[0][column]":      "period",
            "sort[0][direction]":   "desc",
            "length":               1,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            price = float(r.json()["response"]["data"][0]["price"])
            results[key] = price
            log.info(f"  EIA NY {sector}: {price:.2f} ¢/kWh")
        except Exception as e:
            log.warning(f"  EIA {sector} failed: {e} — using fallback")
            results[key] = EIA_FALLBACK[key]

    cache.write_text(json.dumps(results))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — EIA Form 861: Electricity reliability (SAIDI / SAIFI)
# ══════════════════════════════════════════════════════════════════════════════
# Bulk download — no auth required.
# Source: https://www.eia.gov/electricity/data/eia861/
#
# EIA-861 Reliability sheet column layout (0-indexed, after skipping rows 0-1):
#   0  Data Year       4  Ownership
#   1  Utility Number  5-16  IEEE Standard columns (Con Edison does NOT use)
#   2  Utility Name    17  Other Std SAIDI w/ MED  18  Other Std SAIFI w/ MED
#   3  State           20  Other Std SAIDI w/o MED 21  Other Std SAIFI w/o MED
#
# Con Edison (NYC) reports under "Other Standard" (not IEEE Standard).
# MED = Major Event Days.

EIA861_URL      = "https://www.eia.gov/electricity/data/eia861/zip/f8612024.zip"
EIA861_REL_FILE = "Reliability_2024.xlsx"

# Hardcoded fallback from Con Edison 2024 EIA-861 filing
EIA861_FALLBACK = {
    "saidi_with_major_events_min_yr":     27.8,
    "saifi_with_major_events_per_yr":      0.13,
    "saidi_excl_major_events_min_yr":     14.9,
    "saifi_excl_major_events_per_yr":      0.11,
    "caidi_with_major_events_min":       213.7,
}


def get_eia861_reliability() -> dict:
    """Download EIA Form 861 and extract Con Edison SAIDI/SAIFI metrics."""
    cache = CACHE_DIR / "eia861_coned_reliability.json"
    if cache.exists():
        log.info("EIA-861: loading from cache …")
        return json.loads(cache.read_text())

    log.info(f"Downloading EIA Form 861 (~4.5 MB) …")
    try:
        r = requests.get(EIA861_URL, timeout=90, stream=True)
        r.raise_for_status()
        raw = b"".join(r.iter_content(65536))
        log.info(f"  Downloaded {len(raw)//1024} KB")

        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open(EIA861_REL_FILE) as f:
                # Rows 0-1 are group/subgroup headers; row 2 is the column name row
                df = pd.read_excel(f, header=2)

        # Column positions (0-indexed) after row 2 becomes the header:
        # The header row has many duplicate column names like "SAIDI (minutes per year)".
        # Pandas appends .1, .2, .3 ... to deduplicate.
        # Col positions map as follows based on the raw header inspection:
        #   17 → Other Std SAIDI w/ MED  (.3 suffix in deduped pandas headers)
        #   18 → Other Std SAIFI w/ MED
        #   20 → Other Std SAIDI w/o MED
        #   21 → Other Std SAIFI w/o MED
        #   19 → Other Std CAIDI w/ MED

        # Safer: use positional indexing via .iloc
        df = df.reset_index(drop=True)

        # Identify Con Edison row (utility number or name)
        util_num_col  = df.columns[1]   # "Utility Number"
        util_name_col = df.columns[2]   # "Utility Name"

        coned_mask = (
            df[util_num_col].astype(str).str.strip() == str(NYC_UTILITY_NUM)
        ) | (
            df[util_name_col].astype(str).str.contains(
                "Consolidated Edison", case=False, na=False
            )
        )
        coned_rows = df[coned_mask]

        if coned_rows.empty:
            log.warning("  Con Edison not found — using hardcoded fallback values")
            return EIA861_FALLBACK

        row = coned_rows.iloc[0]

        def safe_float(idx: int) -> Optional[float]:
            try:
                v = row.iloc[idx]
                return float(v) if str(v) not in {".", "nan", ""} else None
            except Exception:
                return None

        result = {
            "saidi_with_major_events_min_yr":    safe_float(17),
            "saifi_with_major_events_per_yr":    safe_float(18),
            "caidi_with_major_events_min":       safe_float(19),
            "saidi_excl_major_events_min_yr":    safe_float(20),
            "saifi_excl_major_events_per_yr":    safe_float(21),
        }

        # Fill any None values with fallback
        for k, fallback_v in EIA861_FALLBACK.items():
            if result.get(k) is None:
                log.warning(f"  {k} missing from EIA-861 — using fallback {fallback_v}")
                result[k] = fallback_v

        log.info(f"  SAIDI (w/ major events): {result['saidi_with_major_events_min_yr']:.1f} min/yr")
        log.info(f"  SAIFI (w/ major events): {result['saifi_with_major_events_per_yr']:.2f} events/yr")
        log.info(f"  SAIDI (excl. major):     {result['saidi_excl_major_events_min_yr']:.1f} min/yr")
        cache.write_text(json.dumps(result))
        return result

    except Exception as e:
        log.warning(f"  EIA-861 download failed: {e} — using fallback values")
        return EIA861_FALLBACK


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Derived columns
# ══════════════════════════════════════════════════════════════════════════════

# Monthly kWh for a device running 24/7 (30.44 avg days/month)
def monthly_kwh(watts: int) -> float:
    return round(watts * 24 * 30.44 / 1000, 2)


HARDWARE_PROFILES = {
    # label: (watts, description)
    "mac_mini_m4":    (35,  "Mac Mini M4 — small local model (Llama 3.2 3B)"),
    "workstation_gpu":(350, "GPU workstation — mid model (Mistral 7B / Llama 3 8B)"),
    "dual_gpu":       (900, "Dual GPU server — large model (Llama 3 70B)"),
}


def reliability_tier(saidi: Optional[float]) -> str:
    """
    Classify reliability based on SAIDI total (min/yr) excluding major events.
    Thresholds from EPRI Distribution System Reliability benchmarks.
    """
    if saidi is None:
        return "unknown"
    if saidi < 60:    return "excellent"   # < 1 hr/yr
    if saidi < 120:   return "good"        # 1–2 hr/yr
    if saidi < 300:   return "fair"        # 2–5 hr/yr
    return "poor"                          # > 5 hr/yr


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Assemble
# ══════════════════════════════════════════════════════════════════════════════

COLUMN_ORDER = [
    # Geography
    "zipcode", "borough", "neighborhood", "county", "lat", "lon",

    # Internet access — Census ACS 2022 5-yr (table B28002)
    "census_hh_total",
    "census_hh_any_subscription",     # any internet subscription (incl. cellular)
    "census_hh_fixed_broadband",      # cable / fiber optic / DSL only
    "census_hh_no_internet",          # no internet access at all
    "census_fixed_broadband_pct",     # % HH with fixed broadband (key metric)
    "census_no_internet_pct",         # % HH with no internet

    # FCC Broadband (join manually from broadbandmap.fcc.gov/data-download)
    "fcc_isp_count",           # placeholder — NaN unless FCC data joined
    "fcc_max_dl_mbps",
    "fcc_fiber_available",
    "fcc_cable_available",

    # Electricity cost — EIA
    "utility_name",
    "commercial_rate_cents_kwh",
    "residential_rate_cents_kwh",
    "monthly_cost_mac_mini_m4_usd",
    "monthly_cost_workstation_gpu_usd",
    "monthly_cost_dual_gpu_usd",

    # Electricity reliability — EIA Form 861 2024
    "saidi_with_major_events_min_yr",
    "saifi_with_major_events_per_yr",
    "caidi_with_major_events_min",
    "saidi_excl_major_events_min_yr",
    "saifi_excl_major_events_per_yr",
    "reliability_tier",

    # Metadata
    "data_sources",
]


def build() -> pd.DataFrame:
    # 1. ZIP codes
    zips = get_nyc_zipcodes()

    # 2. Census internet
    census = get_census_internet(set(zips["zipcode"].tolist()))

    # 3. EIA rates
    rates = get_eia_rates()

    # 4. EIA-861 reliability
    reliability = get_eia861_reliability()

    # ── Join ──────────────────────────────────────────────────────────────────
    df = zips.merge(census, on="zipcode", how="left")

    # Electricity — same for all 5 boroughs (all Con Edison)
    df["utility_name"]               = NYC_UTILITY_NAME
    df["commercial_rate_cents_kwh"]  = rates["commercial_rate_cents_kwh"]
    df["residential_rate_cents_kwh"] = rates["residential_rate_cents_kwh"]

    # Monthly cost per hardware profile (using commercial rate)
    rate_per_kwh = rates["commercial_rate_cents_kwh"] / 100
    for label, (watts, _) in HARDWARE_PROFILES.items():
        df[f"monthly_cost_{label}_usd"] = round(monthly_kwh(watts) * rate_per_kwh, 2)

    # Reliability (same for all NYC ZIPs)
    for k, v in reliability.items():
        df[k] = v
    df["reliability_tier"] = reliability_tier(reliability.get("saidi_excl_major_events_min_yr"))

    # FCC placeholder columns (user can join from FCC bulk download)
    for col in ["fcc_isp_count", "fcc_max_dl_mbps", "fcc_fiber_available", "fcc_cable_available"]:
        df[col] = pd.NA

    # Metadata
    df["data_sources"] = (
        "Census ACS 5yr 2022 (B28002); "
        "EIA retail-sales API 2023; "
        "EIA Form 861 2024; "
        "FCC NBM placeholder (join from broadbandmap.fcc.gov/data-download)"
    )

    # Reorder
    extra = [c for c in df.columns if c not in COLUMN_ORDER]
    df = df[[c for c in COLUMN_ORDER if c in df.columns] + extra]
    df = df.sort_values(["borough", "zipcode"]).reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("═" * 64)
    print("  NYC Clinic AI Infrastructure Dataset")
    print("  github.com/Layered-Labs/nyc-clinic-ai-infrastructure")
    print("═" * 64)

    if not EIA_API_KEY:
        print("\n⚠  EIA_API_KEY not set — using hardcoded Oct 2024 NY averages.")
        print("   Get a free key: https://www.eia.gov/opendata/register.php\n")

    df = build()

    out_csv = OUTPUT_DIR / "nyc_clinic_infrastructure.csv"
    df.to_csv(out_csv, index=False)

    print()
    print("─" * 64)
    print(f"  ✓ {len(df)} rows × {len(df.columns)} columns  →  {out_csv}")
    print("─" * 64)
    print()

    # Summary stats
    boros = df["borough"].value_counts().to_dict()
    print("  Boroughs:", boros)

    n_census = df["census_fixed_broadband_pct"].notna().sum()
    avg_bb   = df["census_fixed_broadband_pct"].mean()
    print(f"  Census fixed broadband: {n_census}/{len(df)} ZIPs matched")
    print(f"    NYC avg fixed broadband (cable/fiber/DSL): {avg_bb:.1f}%")

    no_inet  = df[df["census_no_internet_pct"].notna()]["census_no_internet_pct"]
    print(f"    Worst ZIP (no internet): {no_inet.max():.1f}%")
    print(f"    Best ZIP (no internet):  {no_inet.min():.1f}%")

    print(f"\n  Electricity: {df['commercial_rate_cents_kwh'].iloc[0]:.2f} ¢/kWh (commercial)")
    print(f"    Mac Mini M4 (35W):        ${df['monthly_cost_mac_mini_m4_usd'].iloc[0]:.2f}/mo")
    print(f"    GPU workstation (350W):   ${df['monthly_cost_workstation_gpu_usd'].iloc[0]:.2f}/mo")
    print(f"    Dual GPU (900W):          ${df['monthly_cost_dual_gpu_usd'].iloc[0]:.2f}/mo")

    r = df.iloc[0]
    print(f"\n  Reliability ({NYC_UTILITY_NAME}):")
    print(f"    SAIDI w/ major events:  {r['saidi_with_major_events_min_yr']:.1f} min/yr")
    print(f"    SAIDI excl major:       {r['saidi_excl_major_events_min_yr']:.1f} min/yr")
    print(f"    SAIFI w/ major events:  {r['saifi_with_major_events_per_yr']:.2f} outages/yr")
    print(f"    Tier: {r['reliability_tier']}")

    print()
    print("  ⚠  FCC broadband columns (fcc_*) are NaN placeholders.")
    print("     Download NY availability CSV from broadbandmap.fcc.gov/data-download")
    print("     then join to this file on zipcode for complete coverage data.")
    print()

    return df


if __name__ == "__main__":
    main()
