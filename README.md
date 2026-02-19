# NYC Clinic AI Infrastructure

<p align="center">
  <a href="https://layeredlabs.ai">layeredlabs.ai</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/Layered-Labs/nyc-clinic-ai-infrastructure">Dataset</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/spaces/Layered-Labs/nyc-clinic-ai-infra-map">Live Map</a> &nbsp;·&nbsp;
  <a href="https://github.com/Layered-Labs/nyc-clinic-ai-infrastructure">GitHub</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Coverage-311_NYC_ZIPs-green?style=flat-square" />
  <img src="https://img.shields.io/badge/FCC_NBM-Jun_2025-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Census_ACS-2022-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/EIA_Form_861-2024-orange?style=flat-square" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/🤗_Dataset-Layered--Labs-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/🤗_Space-Live_Demo-blue?style=flat-square" />
</p>

> **Can a clinic run local AI? This dataset answers that question for every ZIP code in New York City.**

Published by [Layered Labs](https://layeredlabs.ai), a research lab building open-source AI tools for community health clinics.

---

## Overview

Running AI inference locally on clinic hardware (rather than routing data through cloud APIs) keeps patient data on-premise, reduces cost, and enables functionality in low-connectivity environments. But feasibility depends on infrastructure.

This dataset maps three prerequisites across all 311 NYC ZIP codes:

| Dimension | Source | Vintage |
|-----------|--------|---------|
| Fixed broadband access (household subscription rates) | US Census ACS | 2022 5-yr |
| ISP coverage, max speeds, fiber/cable/copper availability | FCC National Broadband Map | Jun 2025 |
| Commercial electricity rates | EIA Open Data API | 2023 |
| Grid reliability (SAIDI / SAIFI / CAIDI) | EIA Form 861 | 2024 |

---

## Key Findings

### Electricity is not the barrier

Con Edison serves all 5 boroughs with exceptional reliability. Running local AI is inexpensive at NYC commercial rates (17.82 ¢/kWh):

| Hardware | Model tier | Power | Monthly cost |
|----------|-----------|-------|-------------|
| Mac Mini M4 | Llama 3.2 3B | 35W | **$4.56** |
| GPU workstation | Mistral 7B / Llama 3 8B | 350W | **$45.57** |
| Dual-GPU server | Llama 3 70B | 900W | **$117.17** |

Grid reliability: SAIDI of **14.9 min/yr** (excl. major events), top-tier nationally.

### Broadband is the real barrier

Fixed broadband access varies sharply across boroughs:

| Borough | Avg fixed broadband | Avg no internet |
|---------|-------------------|----------------|
| Manhattan | ~83% | ~5% |
| Queens | ~76% | ~10% |
| Brooklyn | ~70% | ~13% |
| Staten Island | ~72% | ~8% |
| **Bronx** | **~66%** | **~14%** |

South Bronx ZIPs (10452, 10454, 10455) sit at **55-63%** fixed broadband, the lowest in the city, while serving some of NYC's most underserved health populations. These are exactly the communities where local, offline-capable AI matters most.

Despite household subscription gaps, **FCC infrastructure coverage is near-universal**: fiber-to-premises is available in 197/197 ZCTAs with FCC data, and the average ZIP has 6 competing ISPs. The gap is a cost and digital equity problem, not an infrastructure one.

---

## Repository Structure

```
nyc-clinic-ai-infrastructure/
├── pipeline/                   # Data collection and processing
│   ├── build_dataset.py        # Step 1: Census ACS + EIA rates + EIA-861
│   ├── join_fcc.py             # Step 2: Join FCC broadband coverage data
│   ├── requirements.txt
│   └── outputs/
│       └── nyc_clinic_infrastructure.csv   # Final dataset (311 rows x 34 cols)
├── explorer/                   # Interactive map (Gradio + Plotly)
│   ├── app.py                  # Deployed at HuggingFace Spaces
│   ├── requirements.txt
│   └── nyc_clinic_infrastructure.csv
└── push_to_hf.py               # Publish dataset and Space to HuggingFace
```

---

## Reproducing the Dataset

All data sources are free and public.

**Prerequisites:**
```bash
pip install -r pipeline/requirements.txt
export EIA_API_KEY=your_key   # free at https://www.eia.gov/opendata/register.php
```

**Step 1: Build base dataset** (Census ACS + EIA rates + EIA-861 reliability):
```bash
python pipeline/build_dataset.py
# Output: pipeline/outputs/nyc_clinic_infrastructure.csv
```

**Step 2: Join FCC broadband data:**
1. Create a free account at [broadbandmap.fcc.gov](https://broadbandmap.fcc.gov/data-download)
2. Download New York state files: `Cable`, `Fiber to the Premises`, `Copper`
3. Place CSVs in `pipeline/requested/`
```bash
python pipeline/join_fcc.py
# Updates: pipeline/outputs/nyc_clinic_infrastructure.csv
```

---

## Dataset Schema

34 columns covering geography, broadband access, ISP coverage, electricity cost, and grid reliability. Full schema in the [HuggingFace dataset card](https://huggingface.co/datasets/Layered-Labs/nyc-clinic-ai-infrastructure).

Key columns:

| Column | Description |
|--------|-------------|
| `census_fixed_broadband_pct` | % households with cable, fiber, or DSL |
| `census_no_internet_pct` | % households with no internet access |
| `fcc_isp_count` | Unique ISPs serving the ZIP |
| `fcc_fiber_available` | Whether fiber-to-premises is available |
| `fcc_max_dl_mbps_practical` | Max download speed, capped at 10 Gbps |
| `monthly_cost_mac_mini_m4_usd` | Monthly electricity cost for a 35W device |
| `saidi_excl_major_events_min_yr` | Grid outage duration per year (minutes) |
| `reliability_tier` | excellent / good / fair / poor |

---

## Citation

```bibtex
@dataset{nyc_clinic_ai_infrastructure_2025,
  title     = {NYC Clinic AI Infrastructure Dataset},
  author    = {{Layered Labs}},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/Layered-Labs/nyc-clinic-ai-infrastructure},
  note      = {FCC NBM Jun-2025, Census ACS 2022, EIA Form 861 2024}
}
```

---

## License

**CC BY 4.0.** Free to use, share, and adapt with attribution.

Underlying data is from US government sources (FCC, Census Bureau, EIA) and is in the public domain.

---

<p align="center">
  Built by <a href="https://layeredlabs.ai">Layered Labs</a>
</p>
