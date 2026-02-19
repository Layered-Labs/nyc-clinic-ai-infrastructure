"""
app.py: NYC Clinic AI Infrastructure Explorer
Gradio + Plotly choropleth map for the NYC Clinic AI Infrastructure Dataset.
"""

import json
import warnings
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

warnings.filterwarnings("ignore")

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv("nyc_clinic_infrastructure.csv", dtype={"zipcode": str})
    df["zipcode"] = df["zipcode"].str.zfill(5)
    # Coerce booleans stored as strings
    for col in ["fcc_fiber_available", "fcc_cable_available", "fcc_copper_available"]:
        if col in df.columns:
            df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0}).astype("Int8")
    return df


def load_geojson(nyc_zips: set) -> dict:
    url = (
        "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON"
        "/master/ny_new_york_zip_codes_geo.min.json"
    )
    geo = requests.get(url, timeout=30).json()
    geo["features"] = [
        f for f in geo["features"]
        if f["properties"]["ZCTA5CE10"] in nyc_zips
    ]
    return geo


df      = load_data()
geojson = load_geojson(set(df["zipcode"].tolist()))

# ── Metric registry ────────────────────────────────────────────────────────────

METRICS = {
    "Fixed Broadband Access (%)": {
        "col":         "census_fixed_broadband_pct",
        "scale":       "RdYlGn",
        "unit":        "%",
        "fmt":         ".1f",
        "high_good":   True,
        "blurb":       "Households with cable, fiber optic, or DSL subscription (Census ACS 2022).",
    },
    "No Internet Access (%)": {
        "col":         "census_no_internet_pct",
        "scale":       "RdYlGn_r",
        "unit":        "%",
        "fmt":         ".1f",
        "high_good":   False,
        "blurb":       "Households with no internet access at all (Census ACS 2022).",
    },
    "Number of ISPs": {
        "col":         "fcc_isp_count",
        "scale":       "Blues",
        "unit":        "ISPs",
        "fmt":         ".0f",
        "high_good":   True,
        "blurb":       "Unique ISPs offering fixed broadband service (FCC NBM Jun-2025).",
    },
    "Max Download Speed (Gbps)": {
        "col":         "fcc_max_dl_mbps_practical",
        "scale":       "Teal",
        "unit":        "Gbps",
        "fmt":         ".0f",
        "high_good":   True,
        "transform":   lambda x: x / 1000,
        "blurb":       "Max advertised download speed, capped at 10 Gbps (FCC NBM Jun-2025).",
    },
    "Fiber Available": {
        "col":         "fcc_fiber_available",
        "scale":       "RdYlGn",
        "unit":        "",
        "fmt":         ".0f",
        "high_good":   True,
        "blurb":       "Whether any fiber-to-premises ISP serves this ZIP (FCC NBM Jun-2025).",
    },
}

BOROUGH_COLORS = {
    "Bronx":         "#e63946",
    "Brooklyn":      "#457b9d",
    "Queens":        "#2a9d8f",
    "Staten Island": "#e9c46a",
    "Manhattan":     "#9b5de5",
}

# ── Plot builders ──────────────────────────────────────────────────────────────

def make_map(metric_name: str) -> go.Figure:
    cfg   = METRICS[metric_name]
    col   = cfg["col"]
    xform = cfg.get("transform", lambda x: x)

    plot_df = df[["zipcode", "borough", "neighborhood", col]].copy()
    plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").apply(
        lambda v: xform(v) if pd.notna(v) else v
    )

    hover = {
        "zipcode":    True,
        "borough":    True,
        col:          f":{cfg['fmt']}",
    }

    fig = px.choropleth_mapbox(
        plot_df,
        geojson=geojson,
        locations="zipcode",
        featureidkey="properties.ZCTA5CE10",
        color=col,
        color_continuous_scale=cfg["scale"],
        range_color=(
            plot_df[col].quantile(0.05),
            plot_df[col].quantile(0.95),
        ),
        mapbox_style="carto-positron",
        zoom=9.8,
        center={"lat": 40.730, "lon": -73.940},
        opacity=0.78,
        hover_name="neighborhood",
        hover_data=hover,
        labels={col: f"{metric_name} ({cfg['unit']})".strip("()")},
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=490,
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title=f"{cfg['unit']}",
            thickness=14,
            len=0.6,
        ),
    )
    return fig


def make_bar(metric_name: str) -> go.Figure:
    cfg   = METRICS[metric_name]
    col   = cfg["col"]
    xform = cfg.get("transform", lambda x: x)

    borough_avg = (
        df.groupby("borough")[col]
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={col: "value"})
        .assign(value=lambda d: d["value"].apply(xform))
        .sort_values("value", ascending=cfg["high_good"])
    )

    fig = px.bar(
        borough_avg,
        x="value",
        y="borough",
        orientation="h",
        color="borough",
        color_discrete_map=BOROUGH_COLORS,
        labels={"value": f"{metric_name}", "borough": ""},
        text=borough_avg["value"].apply(lambda v: f"{v:{cfg['fmt']}}"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=240,
        showlegend=False,
        margin={"r": 60, "t": 10, "l": 10, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=False),
    )
    return fig


def make_stats(metric_name: str) -> str:
    cfg   = METRICS[metric_name]
    col   = cfg["col"]
    xform = cfg.get("transform", lambda x: x)

    vals = pd.to_numeric(df[col], errors="coerce").apply(
        lambda v: xform(v) if pd.notna(v) else v
    )
    valid = vals.dropna()

    if cfg["high_good"]:
        best_idx  = valid.idxmax()
        worst_idx = valid.idxmin()
    else:
        best_idx  = valid.idxmin()
        worst_idx = valid.idxmax()

    def fmt(v):
        return f"{v:{cfg['fmt']}} {cfg['unit']}".strip()

    best_zip  = df.loc[best_idx, "zipcode"]
    best_bor  = df.loc[best_idx, "borough"]
    worst_zip = df.loc[worst_idx, "zipcode"]
    worst_bor = df.loc[worst_idx, "borough"]

    return f"""
### 📊 Stats: {metric_name}

| | |
|---|---|
| **NYC median** | `{fmt(valid.median())}` |
| **NYC average** | `{fmt(valid.mean())}` |
| **Best ZIP** | `{best_zip}` {best_bor}: `{fmt(valid[best_idx])}` |
| **Worst ZIP** | `{worst_zip}` {worst_bor}: `{fmt(valid[worst_idx])}` |
| **ZIPs with data** | `{len(valid)} / 311` |

---
_{cfg['blurb']}_

---
**Electricity (citywide, Con Edison)**
| | |
|---|---|
| SAIDI excl. major events | `14.9 min/yr` |
| Mac Mini M4 (35W, 24/7) | `$4.56 / month` |
| GPU workstation (350W) | `$45.57 / month` |
| Reliability tier | `excellent` |
"""


def update(metric_name: str):
    return make_map(metric_name), make_bar(metric_name), make_stats(metric_name)


# ── Gradio UI ──────────────────────────────────────────────────────────────────

INITIAL_METRIC = "Fixed Broadband Access (%)"

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    title="NYC Clinic AI Infrastructure Explorer",
    css="""
        .metric-card { border-radius: 10px; padding: 12px; }
        footer { display: none !important; }
    """,
) as demo:

    gr.Markdown("""
# 🏥 NYC Clinic AI Infrastructure Explorer

**Research dataset for evaluating on-premise / local AI deployment at clinics across all 311 NYC ZIP codes.**

Combines FCC broadband coverage (Jun 2025), US Census internet access data (ACS 2022),
and EIA electricity cost & reliability data (2024) into a single ZIP-level dataset.

> 📂 [Dataset on Hugging Face](https://huggingface.co/datasets/{{HF_ORG}}/nyc-clinic-ai-infrastructure) &nbsp;·&nbsp;
> 💻 [Source code](https://huggingface.co/datasets/{{HF_ORG}}/nyc-clinic-ai-infrastructure) &nbsp;·&nbsp;
> 📄 CC BY 4.0
""")

    with gr.Row():
        metric_dd = gr.Dropdown(
            label="Explore metric",
            choices=list(METRICS.keys()),
            value=INITIAL_METRIC,
            scale=2,
        )

    with gr.Row():
        with gr.Column(scale=3):
            map_plot = gr.Plot(show_label=False)
        with gr.Column(scale=1, min_width=260):
            stats_md = gr.Markdown()

    with gr.Row():
        bar_plot = gr.Plot(show_label=False)

    gr.Markdown("""
---
### About This Dataset

Running AI inference locally on clinic hardware (rather than calling cloud APIs) can:
- Keep patient data on-premise (PHI compliance)
- Work in low-connectivity environments
- Cost a fraction of cloud inference at scale

This dataset maps the **infrastructure prerequisites** (internet + electricity) by ZIP code
so clinics and researchers can evaluate feasibility at a neighborhood level.

**Key finding:** NYC's electricity grid is excellent for 24/7 inference everywhere (Con Edison SAIDI: 14.9 min/yr).
The real barrier is **broadband access equity**. South Bronx ZIPs with 55-63% fixed broadband
are exactly the communities with the greatest need for reliable, local AI tools.
""")

    # Wire up interactivity
    metric_dd.change(
        fn=update,
        inputs=metric_dd,
        outputs=[map_plot, bar_plot, stats_md],
    )
    demo.load(
        fn=lambda: update(INITIAL_METRIC),
        outputs=[map_plot, bar_plot, stats_md],
    )

if __name__ == "__main__":
    demo.launch()
