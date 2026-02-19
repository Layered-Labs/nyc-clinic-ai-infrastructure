---
license: cc-by-4.0
language:
- en
tags:
- healthcare
- infrastructure
- broadband
- electricity
- new-york-city
- local-ai
- on-premise-ai
pretty_name: NYC Clinic AI Infrastructure Dataset
size_categories:
- n<1K
task_categories:
- other
---

# 🏥 NYC Clinic AI Infrastructure Dataset

> **Can clinics run local AI? This dataset answers that question for every ZIP code in New York City.**

Published by a nonprofit researching open-source, on-premise AI deployment for community health clinics.

| | |
|---|---|
| **Coverage** | 311 ZIP codes · all 5 NYC boroughs |
| **Dimensions** | Broadband access · Electricity cost · Grid reliability |
| **Data vintage** | FCC NBM Jun 2025 · Census ACS 2022 · EIA Form 861 2024 |
| **License** | CC BY 4.0 |
| **Explore** | [🗺️ Interactive Map](https://huggingface.co/spaces/{{HF_ORG}}/nyc-clinic-ai-infra-map) |

---

## Why This Dataset Exists

Running AI models locally on a clinic's own hardware can reduce costs, eliminate protected health information (PHI) transmission to third parties, and provide AI functionality even in low-connectivity environments. Whether a clinic can viably do this depends on three infrastructure questions:

1. **Internet connectivity:** reliable broadband is needed for model updates and telemetry, even when inference runs fully offline
2. **Electricity cost:** inference hardware running 24/7 adds to operational costs
3. **Grid reliability:** power outages can corrupt model state and interrupt patient-facing tools

This dataset was built to help clinics, researchers, and policymakers evaluate where local AI deployment is most feasible and where infrastructure gaps create barriers.

---

## Key Findings

### 🔌 Electricity: A Non-Issue Citywide

Con Edison serves all 5 boroughs with exceptional reliability:

| Metric | Value |
|--------|-------|
| SAIDI (excl. major events) | **14.9 min/year** |
| SAIDI (incl. major events) | **27.8 min/year** |
| SAIFI | **0.13 outages/year** |
| Reliability tier | **Excellent** (EPRI benchmark) |

Running a local AI model is also inexpensive at NYC commercial rates (17.82 ¢/kWh):

| Hardware | Power | Monthly cost |
|----------|-------|-------------|
| Mac Mini M4 (Llama 3.2 3B) | 35W | **$4.56/mo** |
| GPU workstation (Mistral 7B) | 350W | **$45.57/mo** |
| Dual-GPU server (Llama 3 70B) | 900W | **$117.17/mo** |

### 📡 Broadband: The Real Barrier

Fixed broadband (cable/fiber/DSL) access varies dramatically across boroughs:

| Borough | Avg Fixed Broadband % | Avg No-Internet % |
|---------|-----------------------|-------------------|
| Manhattan | ~83% | ~5% |
| Queens | ~76% | ~10% |
| Brooklyn | ~70% | ~13% |
| Staten Island | ~72% | ~8% |
| **Bronx** | **~66%** | **~14%** |

> South Bronx ZIPs (10452, 10454, 10455) have the city's lowest fixed broadband rates at 55-63%, while being home to some of NYC's most underserved health populations. This is where local, offline-capable AI matters most.

### 🌐 Broadband Infrastructure

Despite household subscription gaps, **FCC infrastructure coverage is near-universal**:
- Fiber-to-premises available: **197/197 ZCTAs with FCC data**
- Cable available: **196/197 ZCTAs**
- Average ISPs per ZIP: **6.0**

The gap is between *infrastructure availability* and *household subscription*: a cost and digital equity problem, not an infrastructure one.

---

## Dataset Schema

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `zipcode` | string | 5-digit ZIP code | GeoNames/TIGER |
| `borough` | string | NYC borough | GeoNames/TIGER |
| `neighborhood` | string | Neighborhood / place name | GeoNames/TIGER |
| `lat`, `lon` | float | ZIP centroid coordinates | GeoNames/TIGER |
| `census_hh_total` | int | Total households | Census ACS 5yr 2022 |
| `census_hh_any_subscription` | int | HH with any internet subscription | Census ACS B28002_002E |
| `census_hh_fixed_broadband` | int | HH with cable/fiber/DSL | Census ACS B28002_007E |
| `census_hh_no_internet` | int | HH with no internet access | Census ACS B28002_013E |
| `census_fixed_broadband_pct` | float | % HH with fixed broadband | Derived |
| `census_no_internet_pct` | float | % HH with no internet | Derived |
| `fcc_isp_count` | int | Unique ISPs offering service in ZIP | FCC NBM Jun-2025 |
| `fcc_max_dl_mbps` | int | Max advertised download speed (Mbps) | FCC NBM Jun-2025 |
| `fcc_max_dl_mbps_practical` | int | Max DL speed capped at 10 Gbps | FCC NBM (derived) |
| `fcc_max_ul_mbps` | int | Max advertised upload speed (Mbps) | FCC NBM Jun-2025 |
| `fcc_max_ul_mbps_practical` | int | Max UL speed capped at 10 Gbps | FCC NBM (derived) |
| `fcc_location_count` | int | Unique addressable locations with service | FCC NBM Jun-2025 |
| `fcc_cable_available` | bool | Cable (HFC/DOCSIS) ISP present | FCC NBM Jun-2025 |
| `fcc_fiber_available` | bool | Fiber-to-premises ISP present | FCC NBM Jun-2025 |
| `fcc_copper_available` | bool | Copper/DSL ISP present | FCC NBM Jun-2025 |
| `utility_name` | string | Primary electric utility | EIA Form 861 2024 |
| `commercial_rate_cents_kwh` | float | Commercial electricity rate (¢/kWh) | EIA API 2023 |
| `residential_rate_cents_kwh` | float | Residential electricity rate (¢/kWh) | EIA API 2023 |
| `monthly_cost_mac_mini_m4_usd` | float | Monthly cost: 35W device, 24/7 ($) | Derived |
| `monthly_cost_workstation_gpu_usd` | float | Monthly cost: 350W GPU, 24/7 ($) | Derived |
| `monthly_cost_dual_gpu_usd` | float | Monthly cost: 900W dual-GPU, 24/7 ($) | Derived |
| `saidi_with_major_events_min_yr` | float | SAIDI incl. major events (min/yr) | EIA Form 861 2024 |
| `saidi_excl_major_events_min_yr` | float | SAIDI excl. major events (min/yr) | EIA Form 861 2024 |
| `saifi_with_major_events_per_yr` | float | SAIFI incl. major events (events/yr) | EIA Form 861 2024 |
| `saifi_excl_major_events_per_yr` | float | SAIFI excl. major events (events/yr) | EIA Form 861 2024 |
| `caidi_with_major_events_min` | float | CAIDI incl. major events (min/outage) | EIA Form 861 2024 |
| `reliability_tier` | string | excellent / good / fair / poor | EPRI benchmarks |
| `data_sources` | string | Full provenance string | n/a |

**Notes:**
- `fcc_*` columns cover 197 of 311 ZIPs (ZIPs without Census ZCTAs have NaN broadband columns; ZIPs without residential population have NaN census columns)
- `fcc_max_dl_mbps` includes carrier-grade dark fiber (e.g. Zayo at 1.2 Tbps, business-only); use `fcc_max_dl_mbps_practical` (capped at 10 Gbps) for clinic-relevant speeds
- Electricity metrics are uniform citywide (all ZIPs served by Consolidated Edison Co-NY Inc)

---

## Reproducing the Dataset

All data sources are free and public. The pipeline is fully automated.

**Prerequisites:**
```bash
pip install -r requirements.txt
export EIA_API_KEY=your_key   # free at https://www.eia.gov/opendata/register.php
```

**Step 1: Base dataset** (Census ACS + EIA rates + EIA-861 reliability):
```bash
python build_dataset.py
# Output: outputs/nyc_clinic_infrastructure.csv
```

**Step 2: Add FCC broadband data:**
1. Create a free account at [broadbandmap.fcc.gov](https://broadbandmap.fcc.gov/data-download)
2. Download for **New York** state: `Cable`, `Fiber to the Premises`, `Copper` (Fixed Broadband)
3. Place CSVs in `requested/`
```bash
python join_fcc.py
# Updates: outputs/nyc_clinic_infrastructure.csv
```

**Data sources accessed programmatically (no account needed):**
- US Census ACS 5-year API: `api.census.gov`
- EIA Open Data API: `api.eia.gov`
- EIA Form 861 bulk download: `eia.gov/electricity/data/eia861`
- Census ZCTA-Tract crosswalk: `www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520`

---

## Citation

If you use this dataset in research, please cite:

```bibtex
@dataset{nyc_clinic_ai_infrastructure_2025,
  title     = {NYC Clinic AI Infrastructure Dataset},
  author    = {{{{HF_ORG}}}},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/{{HF_ORG}}/nyc-clinic-ai-infrastructure},
  note      = {FCC NBM Jun-2025, Census ACS 2022, EIA Form 861 2024}
}
```

---

## License

**CC BY 4.0.** You are free to share and adapt this dataset for any purpose, provided you give appropriate credit.

Underlying data is from US government sources (FCC, Census Bureau, EIA) and is in the public domain.
