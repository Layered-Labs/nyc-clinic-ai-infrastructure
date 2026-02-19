#!/usr/bin/env python3
"""
join_fcc.py
===========
Joins FCC National Broadband Map location-level data to the existing
nyc_clinic_infrastructure.csv dataset at the ZIP (ZCTA) level.

Inputs:
  requested/bdc_36_Cable_fixed_broadband_J25_17feb2026.csv        (371 MB)
  requested/bdc_36_Copper_fixed_broadband_J25_17feb2026.csv       (27 MB)
  requested/bdc_36_FibertothePremises_fixed_broadband_J25_17feb2026.csv (565 MB)
  cache/ny_tract_to_zcta.json   (pre-built from Census crosswalk)
  outputs/nyc_clinic_infrastructure.csv

Method:
  1. For each FCC CSV, read in chunks and filter to NYC county prefixes.
     Census block_geoid[:11] = tract GEOID → map to ZCTA via crosswalk.
  2. Aggregate per ZCTA:
       fcc_isp_count           unique brand names across all 3 technologies
       fcc_max_dl_mbps         max advertised download speed
       fcc_max_ul_mbps         max advertised upload speed
       fcc_cable_available     True if cable ISP present
       fcc_fiber_available     True if fiber ISP present
       fcc_copper_available    True if copper/DSL ISP present
       fcc_location_count      unique location_ids with any service
  3. Join to existing CSV, overwrite NaN FCC placeholder columns.

Output: outputs/nyc_clinic_infrastructure.csv (updated in-place)
"""

import json
import logging
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

FCC_DIR    = Path("requested")
CACHE_DIR  = Path("cache")
OUTPUT_CSV = Path("outputs") / "nyc_clinic_infrastructure.csv"

FCC_FILES = {
    "cable": FCC_DIR / "bdc_36_Cable_fixed_broadband_J25_17feb2026.csv",
    "fiber": FCC_DIR / "bdc_36_FibertothePremises_fixed_broadband_J25_17feb2026.csv",
    "copper": FCC_DIR / "bdc_36_Copper_fixed_broadband_J25_17feb2026.csv",
}

# NYC county FIPS prefixes (first 5 chars of 15-digit block_geoid)
NYC_COUNTY_PREFIXES = {"36005", "36047", "36061", "36081", "36085"}

CHUNK_SIZE = 200_000


# ── Step 1: Load tract → ZCTA mapping ─────────────────────────────────────────

def load_tract_zcta_map() -> dict:
    path = CACHE_DIR / "ny_tract_to_zcta.json"
    if not path.exists():
        raise FileNotFoundError(
            "cache/ny_tract_to_zcta.json not found. "
            "Run build_dataset.py first to generate the Census crosswalk."
        )
    mapping = json.loads(path.read_text())
    log.info(f"Loaded {len(mapping):,} NY tract→ZCTA pairs")
    return mapping


# ── Step 2: Process each FCC CSV ──────────────────────────────────────────────

def process_fcc_file(path: Path, tech_label: str, tract_map: dict) -> pd.DataFrame:
    """
    Read one FCC broadband CSV, filter to NYC locations,
    map block_geoid → ZCTA, return per-row DataFrame with zipcode column.
    """
    log.info(f"Processing {path.name} ({path.stat().st_size // 1_048_576} MB) …")

    chunks = []
    total_rows   = 0
    nyc_rows     = 0
    mapped_rows  = 0

    reader = pd.read_csv(
        path,
        usecols=["brand_name", "location_id",
                 "max_advertised_download_speed",
                 "max_advertised_upload_speed",
                 "block_geoid"],
        dtype={
            "brand_name": str,
            "location_id": str,
            "max_advertised_download_speed": "Int32",
            "max_advertised_upload_speed":   "Int32",
            "block_geoid": str,
        },
        chunksize=CHUNK_SIZE,
    )

    for chunk in tqdm(reader, desc=f"  {tech_label}", unit="chunk"):
        total_rows += len(chunk)

        # Filter to NYC counties (first 5 chars of block_geoid)
        chunk["county_prefix"] = chunk["block_geoid"].str[:5]
        nyc = chunk[chunk["county_prefix"].isin(NYC_COUNTY_PREFIXES)].copy()
        nyc_rows += len(nyc)

        if nyc.empty:
            continue

        # Map to ZCTA via tract (first 11 chars of block_geoid)
        nyc["tract"] = nyc["block_geoid"].str[:11]
        nyc["zipcode"] = nyc["tract"].map(tract_map)

        # Drop rows without a ZCTA match
        nyc = nyc.dropna(subset=["zipcode"])
        mapped_rows += len(nyc)

        nyc["tech"] = tech_label
        chunks.append(nyc[["zipcode", "brand_name", "location_id",
                            "max_advertised_download_speed",
                            "max_advertised_upload_speed", "tech"]])

    log.info(f"  Total: {total_rows:,} rows → {nyc_rows:,} NYC → {mapped_rows:,} ZCTA-mapped")

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


# ── Step 3: Aggregate per ZCTA ────────────────────────────────────────────────

def aggregate_by_zcta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ZCTA FCC metrics from location-level data."""
    log.info("Aggregating by ZCTA …")

    # Per-ZCTA aggregations
    agg = df.groupby("zipcode").agg(
        fcc_isp_count           = ("brand_name",                   "nunique"),
        fcc_max_dl_mbps         = ("max_advertised_download_speed", "max"),
        fcc_max_ul_mbps         = ("max_advertised_upload_speed",   "max"),
        fcc_location_count      = ("location_id",                   "nunique"),
    ).reset_index()

    # Boolean tech availability per ZCTA
    for tech in ["cable", "fiber", "copper"]:
        col = f"fcc_{tech}_available"
        zips_with_tech = df[df["tech"] == tech]["zipcode"].unique()
        agg[col] = agg["zipcode"].isin(zips_with_tech)

    agg["fcc_max_dl_mbps"] = agg["fcc_max_dl_mbps"].astype("Int32")
    agg["fcc_max_ul_mbps"] = agg["fcc_max_ul_mbps"].astype("Int32")

    log.info(f"  Aggregated {len(agg)} ZCTAs")
    log.info(f"  Cable: {agg['fcc_cable_available'].sum()} ZCTAs")
    log.info(f"  Fiber: {agg['fcc_fiber_available'].sum()} ZCTAs")
    log.info(f"  Copper: {agg['fcc_copper_available'].sum()} ZCTAs")
    log.info(f"  Max ISP count in any ZCTA: {agg['fcc_isp_count'].max()}")
    log.info(f"  Max DL speed: {agg['fcc_max_dl_mbps'].max()} Mbps")

    return agg


# ── Step 4: Join to existing dataset ──────────────────────────────────────────

def join_and_save(fcc_agg: pd.DataFrame):
    log.info(f"Joining to {OUTPUT_CSV} …")
    base = pd.read_csv(OUTPUT_CSV, dtype={"zipcode": str})
    base["zipcode"] = base["zipcode"].str.zfill(5)

    # Drop old placeholder columns
    drop_cols = [c for c in base.columns if c.startswith("fcc_")]
    base = base.drop(columns=drop_cols)

    # Merge
    merged = base.merge(fcc_agg, on="zipcode", how="left")

    # Place FCC columns right after census columns (before utility columns)
    fcc_cols = [c for c in merged.columns if c.startswith("fcc_")]
    other_cols = [c for c in merged.columns if not c.startswith("fcc_")]

    # Insert FCC cols after census_no_internet_pct
    insert_after = "census_no_internet_pct"
    if insert_after in other_cols:
        idx = other_cols.index(insert_after) + 1
        final_order = other_cols[:idx] + fcc_cols + other_cols[idx:]
    else:
        final_order = other_cols + fcc_cols

    merged = merged[final_order]
    merged.to_csv(OUTPUT_CSV, index=False)

    matched = merged["fcc_isp_count"].notna().sum()
    log.info(f"  Matched FCC data for {matched}/{len(merged)} ZIPs")
    log.info(f"  Saved → {OUTPUT_CSV}")
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("═" * 60)
    print("  FCC Broadband Data Join")
    print("  Data as of: Jun 30, 2025 (last updated 2/17/26)")
    print("═" * 60)

    tract_map = load_tract_zcta_map()

    # Process all 3 technology files
    all_dfs = []
    for tech_label, path in FCC_FILES.items():
        if not path.exists():
            log.warning(f"  {path.name} not found — skipping {tech_label}")
            continue
        df = process_fcc_file(path, tech_label, tract_map)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        log.error("No FCC data processed. Check files in requested/")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Combined: {len(combined):,} NYC location rows across all technologies")

    fcc_agg = aggregate_by_zcta(combined)
    merged  = join_and_save(fcc_agg)

    # Summary
    print()
    print("─" * 60)
    print(f"  ✓ {len(merged)} rows × {len(merged.columns)} columns  →  {OUTPUT_CSV}")
    print("─" * 60)
    print()
    print("  FCC coverage summary (NYC ZCTAs):")
    print(f"    ZCTAs with any FCC data:  {merged['fcc_isp_count'].notna().sum()}")
    print(f"    Cable available:          {merged['fcc_cable_available'].sum()}")
    print(f"    Fiber available:          {merged['fcc_fiber_available'].sum()}")
    print(f"    Copper/DSL available:     {merged['fcc_copper_available'].sum()}")
    print(f"    Avg ISP count per ZCTA:   {merged['fcc_isp_count'].mean():.1f}")
    print(f"    Max download speed:       {merged['fcc_max_dl_mbps'].max()} Mbps")
    print()

    # Show worst-connected ZIPs (most relevant for clinic feasibility)
    worst = (
        merged[merged["fcc_isp_count"].notna()]
        .nsmallest(10, "fcc_isp_count")
        [["zipcode", "borough", "neighborhood",
          "fcc_isp_count", "fcc_max_dl_mbps",
          "fcc_fiber_available", "fcc_cable_available",
          "census_no_internet_pct"]]
    )
    print("  Lowest ISP-count ZIPs (clinic connectivity risk):")
    print(worst.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
