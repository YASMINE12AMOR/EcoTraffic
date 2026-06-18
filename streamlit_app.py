import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml.env_model import (
    FEATURES,
    TARGET,
    evaluate,
    feature_importance,
    load_data,
    load_model,
    save_model,
    train,
)


ROOT_DIR = Path(__file__).parent
MODEL_IMAGE_PATH = ROOT_DIR / "ml" / "models" / "env_model_results.png"
MODELS_COMPARISON_PATH = ROOT_DIR / "ml" / "models" / "models_comparison.json"
TRAFFIC_CLEANED_PATH = ROOT_DIR / "ecotraffic" / "data" / "03_primary" / "traffic_cleaned.csv"

POLLUTANTS = [
    "nitrogen_dioxide",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "ozone",
    "sulphur_dioxide",
]

RENNES_ZONES = pd.DataFrame(
    [
        {"zone": "Centre", "lat": 48.1173, "lon": -1.6778, "weight": 1.00},
        {"zone": "Villejean", "lat": 48.1215, "lon": -1.7041, "weight": 0.92},
        {"zone": "Beaulieu", "lat": 48.1162, "lon": -1.6387, "weight": 0.86},
        {"zone": "Cleunay", "lat": 48.1019, "lon": -1.7046, "weight": 0.90},
        {"zone": "Gare", "lat": 48.1039, "lon": -1.6727, "weight": 1.08},
        {"zone": "Longs-Champs", "lat": 48.1345, "lon": -1.6357, "weight": 0.82},
        {"zone": "Brequigny", "lat": 48.0866, "lon": -1.6854, "weight": 0.88},
    ]
)


st.set_page_config(
    page_title="EcoTraffic Dashboard",
    page_icon="ET",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F8FAFB;
        --panel: #FFFFFF;
        --green-dark: #0D7F5C;
        --green-dark-hover: #0A6B4E;
        --green: #12A86D;
        --green-light: #E8F7F2;
        --green-gradient: linear-gradient(135deg, #12A86D 0%, #0D7F5C 100%);
        --blue: #2563EB;
        --blue-light: #F0F4FF;
        --orange: #D97706;
        --orange-light: #FEF3E8;
        --red: #DC2626;
        --text: #0F1419;
        --text-soft: #6B7280;
        --text-muted: #9CA3AF;
        --border: rgba(0,0,0,0.05);
        --radius: 16px;
        --radius-sm: 12px;
        --shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 24px rgba(0, 0, 0, 0.12);
    }

    * { font-family: 'Inter', sans-serif; box-sizing: border-box; }

    .stApp { background: var(--bg); color: var(--text); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #FFFFFF 0%, #FAFBFC 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: var(--green-gradient) !important;
        border: none !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px 18px !important;
        transition: all 0.3s ease !important;
        width: 100%;
        box-shadow: 0 2px 8px rgba(18, 168, 109, 0.15) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--green-dark-hover) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(18, 168, 109, 0.25) !important;
    }

    /* ── Sidebar logo ── */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
        padding-bottom: 18px;
        border-bottom: 1px solid var(--border);
    }
    .sidebar-logo-icon {
        width: 44px; height: 44px;
        background: var(--green-gradient);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: 800; color: #FFFFFF;
        flex-shrink: 0;
        letter-spacing: -0.5px;
        box-shadow: 0 4px 12px rgba(18, 168, 109, 0.2);
    }
    .sidebar-logo-name {
        font-size: 20px !important;
        font-weight: 800 !important;
        color: var(--green-dark) !important;
        line-height: 1;
    }
    .sidebar-section-label {
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: var(--text-muted) !important;
        margin: 20px 0 10px 2px;
        display: block;
    }
    .sidebar-promo {
        background: var(--green-gradient);
        border-radius: 16px;
        padding: 20px;
        margin-top: 28px;
        box-shadow: 0 8px 20px rgba(18, 168, 109, 0.15);
    }
    .sidebar-promo-title {
        font-size: 15px; font-weight: 700; color: #FFFFFF !important;
        margin-bottom: 8px;
    }
    .sidebar-promo-text {
        font-size: 13px; color: rgba(255,255,255,0.75) !important;
        line-height: 1.6;
    }

    /* ── KPI cards ── */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 16px;
        margin-bottom: 28px;
    }
    .kpi-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px 20px;
        position: relative;
        box-shadow: var(--shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 140px;
        display: flex; flex-direction: column; justify-content: space-between;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--green-gradient);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .kpi-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-4px);
        border-color: rgba(18, 168, 109, 0.15);
    }
    .kpi-card:hover::before {
        opacity: 1;
    }
    .kpi-card.featured {
        background: var(--green-gradient);
        border-color: transparent;
        box-shadow: 0 8px 20px rgba(18, 168, 109, 0.2);
    }
    .kpi-card.featured::before {
        opacity: 0;
    }
    .kpi-top { display: flex; justify-content: space-between; align-items: flex-start; }
    .kpi-label { font-size: 12px; font-weight: 600; color: var(--text-soft); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-card.featured .kpi-label { color: rgba(255,255,255,0.7); }
    .kpi-value {
        font-size: 36px; font-weight: 800; color: var(--text);
        line-height: 1; margin-bottom: 10px;
    }
    .kpi-card.featured .kpi-value { color: #FFFFFF; }
    .kpi-value small { font-size: 15px; font-weight: 500; opacity: 0.6; }
    .kpi-badge {
        display: inline-flex; align-items: center; gap: 5px;
        font-size: 11px; font-weight: 700; padding: 4px 12px;
        border-radius: 20px; background: var(--green-light); color: var(--green);
        max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .kpi-card.featured .kpi-badge { background: rgba(255,255,255,0.2); color: rgba(255,255,255,0.95); }
    .kpi-arrow {
        width: 28px; height: 28px; border-radius: 50%;
        border: 1.5px solid var(--border);
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; color: var(--text-soft); flex-shrink: 0;
    }
    .kpi-card.featured .kpi-arrow { border-color: rgba(255,255,255,0.25); color: rgba(255,255,255,0.8); }

    /* ── Page header ── */
    .page-header {
        display: flex; justify-content: space-between; align-items: flex-end;
        margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid var(--border);
    }
    .page-title { font-size: 32px !important; font-weight: 800 !important; color: var(--green-dark) !important; margin: 0 !important; }
    .page-subtitle { color: var(--text-soft); font-size: 15px; margin: 6px 0 0 !important; }
    .header-badges { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .hbadge {
        font-size: 12px; border-radius: 10px; padding: 8px 14px; font-weight: 600;
        border: 1px solid var(--border); background: var(--panel); color: var(--text-soft);
        transition: all 0.2s ease;
    }
    .hbadge:hover {
        border-color: var(--border);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .hbadge.green { background: var(--green-light); color: var(--green); border-color: transparent; font-weight: 700; }
    .hbadge.status-normal  { background: var(--green-light); color: var(--green); border: none; font-weight: 700; }
    .hbadge.status-vigilance { background: var(--orange-light); color: var(--orange); border: none; font-weight: 700; }
    .hbadge.status-critique  { background: #FEE2E2; color: var(--red); border: none; font-weight: 700; }

    /* ── Inner metric cards (tabs) ── */
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 24px;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(18, 168, 109, 0.1);
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--text-soft) !important; font-size: 12px !important;
        font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--green-dark) !important; font-size: 28px !important; font-weight: 800 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: var(--green) !important; }
    div[data-testid="stMetricDelta"] svg { fill: var(--green) !important; }

    /* ── Small cards ── */
    .small-card {
        border: 1px solid var(--border); border-radius: var(--radius-sm);
        padding: 18px 20px; background: var(--panel); min-height: 85px;
        box-shadow: var(--shadow); margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .small-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: var(--green-gradient);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .small-card:hover {
        border-color: rgba(18, 168, 109, 0.2);
        box-shadow: var(--shadow-md);
        transform: translateX(2px);
    }
    .small-card:hover::before {
        opacity: 1;
    }
    .small-card strong {
        display: block; font-size: 12px; font-weight: 800; margin-bottom: 8px;
        color: var(--green); text-transform: uppercase; letter-spacing: 0.8px;
    }
    .small-card span { color: var(--text-soft); font-size: 14px; line-height: 1.6; }

    .section-note { color: var(--text-muted); margin-top: -4px; margin-bottom: 16px; font-size: 14px; font-weight: 500; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: transparent !important;
        border-bottom: 2px solid var(--border) !important; padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border: none; border-bottom: 3px solid transparent; border-radius: 0;
        background: transparent; padding: 12px 18px;
        color: var(--text-soft) !important; font-size: 14px; font-weight: 600; margin-bottom: -2px;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--green) !important;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-bottom: 3px solid var(--green) !important;
        color: var(--green) !important; font-weight: 800 !important;
    }

    h1, h2, h3, h4, h5, h6 { color: var(--text); font-weight: 700; }
    h1 { color: var(--green-dark); }
    h2 { color: var(--green-dark); font-size: 24px !important; margin-bottom: 16px !important; }
    p, span, label { color: var(--text); }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(18, 168, 109, 0.05); border-radius: 3px; }
    ::-webkit-scrollbar-thumb { background: rgba(18, 168, 109, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(18, 168, 109, 0.5); }

    div[data-testid="stDownloadButton"] > button {
        background: var(--green-gradient) !important;
        border: none !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 10px 18px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(18, 168, 109, 0.2) !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(18, 168, 109, 0.3) !important;
    }

    .stAlert {
        background: linear-gradient(135deg, var(--green-light) 0%, rgba(18, 168, 109, 0.05) 100%) !important;
        border: 1px solid rgba(18, 168, 109, 0.15) !important;
        border-left: 4px solid var(--green) !important;
        color: var(--text) !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: 0 2px 8px rgba(18, 168, 109, 0.08) !important;
    }

    /* ── Multiselect tags ── */
    [data-baseweb="tag"] {
        background: var(--green-gradient) !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(18, 168, 109, 0.15) !important;
    }
    [data-baseweb="tag"] span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 12px !important;
    }
    [data-baseweb="tag"] [role="presentation"] {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def get_data() -> pd.DataFrame:
    """Recharge les données environnement toutes les heures (rythme DAG Airflow @hourly)."""
    df = load_data()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


@st.cache_data(ttl=3600)
def get_traffic_data() -> pd.DataFrame:
    """Recharge le CSV trafic toutes les heures (rythme DAG Airflow @hourly)."""
    if not TRAFFIC_CLEANED_PATH.exists():
        return pd.DataFrame()

    traffic = pd.read_csv(TRAFFIC_CLEANED_PATH)
    if "Date/Heure" not in traffic.columns:
        return pd.DataFrame()

    traffic["Date/Heure"] = pd.to_datetime(traffic["Date/Heure"], errors="coerce")
    traffic = traffic.dropna(subset=["Date/Heure"]).sort_values("Date/Heure")
    return traffic


@st.cache_data(ttl=3600)
def get_model_outputs(df: pd.DataFrame):
    """Ré-entraîne le modèle XGBoost sur les nouvelles données toutes les heures."""
    model, _x_train, x_test, _y_train, y_test = train(df)
    metrics = evaluate(model, x_test, y_test)
    importance = feature_importance(model)
    save_model(model)
    return model, metrics, importance, x_test, y_test


@st.cache_data(ttl=3600)
def get_saved_model():
    """Recharge le modèle sérialisé toutes les heures."""
    try:
        return load_model()
    except Exception:
        return None


def normalize(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_value) / (max_value - min_value)


def compute_traffic_score(traffic_df: pd.DataFrame) -> pd.DataFrame:
    traffic = traffic_df.copy()
    required = [
        "Date/Heure",
        "Vitesse Moyenne (km/h)",
        "Vitesse Max (km/h)",
        "Temps de Trajet (s)",
        "Statut Trafic",
    ]
    if traffic.empty or any(column not in traffic.columns for column in required):
        return pd.DataFrame()

    avg_speed = pd.to_numeric(traffic["Vitesse Moyenne (km/h)"], errors="coerce")
    max_speed = pd.to_numeric(traffic["Vitesse Max (km/h)"], errors="coerce")
    travel_time = pd.to_numeric(traffic["Temps de Trajet (s)"], errors="coerce")
    status = traffic["Statut Trafic"].astype(str).str.lower().str.strip()

    speed_pressure = (1 - (avg_speed / max_speed.replace(0, np.nan))).clip(0, 1).fillna(0)
    status_pressure = status.map(
        {
            "freeflow": 0,
            "heavy": 65,
            "congested": 100,
            "unknown": 35,
        }
    ).fillna(35)

    traffic["traffic_real_score"] = (
        status_pressure * 0.50
        + speed_pressure * 100 * 0.35
        + normalize(travel_time.fillna(travel_time.median())) * 100 * 0.15
    ).clip(0, 100)
    traffic["datetime_hour"] = traffic["Date/Heure"].dt.floor("h")

    return (
        traffic.groupby("datetime_hour", as_index=False)
        .agg(
            traffic_real_score=("traffic_real_score", "mean"),
            avg_speed=("Vitesse Moyenne (km/h)", "mean"),
            routes_observed=("Route", "nunique"),
        )
        .sort_values("datetime_hour")
    )


def add_scores(df: pd.DataFrame, traffic_df: pd.DataFrame | None = None) -> pd.DataFrame:
    scored = df.copy()
    scored["pollution_score"] = (
        normalize(scored["nitrogen_dioxide"]) * 0.38
        + normalize(scored["pm10"]) * 0.24
        + normalize(scored["pm2_5"]) * 0.24
        + normalize(scored["carbon_monoxide"]) * 0.14
    ) * 100

    rush_hour = scored["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    rain_pressure = scored["rain_flag"].astype(int)
    wind_relief = normalize(scored["wind_kmh"])
    scored["traffic_proxy_score"] = (
        rush_hour * 45
        + rain_pressure * 22
        + normalize(scored["carbon_monoxide"]) * 25
        + (1 - wind_relief) * 8
    ).clip(0, 100)

    scored["traffic_score"] = scored["traffic_proxy_score"]
    scored["traffic_source"] = "proxy"
    if traffic_df is not None and not traffic_df.empty:
        traffic_hourly = compute_traffic_score(traffic_df)
        if not traffic_hourly.empty:
            scored["datetime_hour"] = scored["datetime"].dt.floor("h")
            scored = scored.merge(traffic_hourly, on="datetime_hour", how="left")
            if scored["traffic_real_score"].notna().any():
                scored["traffic_score"] = scored["traffic_real_score"].fillna(scored["traffic_proxy_score"])
                scored["traffic_source"] = np.where(
                    scored["traffic_real_score"].notna(),
                    "trafic reel",
                    "proxy",
                )
            else:
                latest_traffic = traffic_hourly.sort_values("datetime_hour").iloc[-1]
                scored["traffic_score"] = float(latest_traffic["traffic_real_score"])
                scored["traffic_source"] = "trafic reel recent"
                scored["avg_speed"] = float(latest_traffic["avg_speed"])
                scored["routes_observed"] = int(latest_traffic["routes_observed"])
        else:
            scored["traffic_real_score"] = np.nan
            scored["avg_speed"] = np.nan
            scored["routes_observed"] = np.nan

    scored["ecotraffic_score"] = (
        scored["pollution_score"] * 0.62 + scored["traffic_score"] * 0.38
    ).clip(0, 100)
    scored["risk_level"] = pd.cut(
        scored["ecotraffic_score"],
        bins=[-1, 35, 65, 100],
        labels=["Normal", "Vigilance", "Critique"],
    )
    return scored


def get_default_value(df: pd.DataFrame, column: str, fallback: float) -> float:
    if column not in df.columns or df[column].dropna().empty:
        return fallback
    return float(df[column].median())


def score_color(score: float) -> str:
    if score >= 65:
        return "#ff5d5d"
    if score >= 35:
        return "#ffb84d"
    return "#39d98a"


def score_label(score: float) -> str:
    if score >= 65:
        return "Critique"
    if score >= 35:
        return "Vigilance"
    return "Normal"


def gauge(title: str, value: float, suffix: str = "") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 34, "color": "#111827"}},
            title={"text": title, "font": {"size": 15, "color": "#6B7280"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(0,0,0,0.25)"},
                "bar": {"color": score_color(value), "thickness": 0.30},
                "bgcolor": "#FFFFFF",
                "borderwidth": 1,
                "bordercolor": "rgba(0,0,0,0.10)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(12, 175, 109, 0.10)"},
                    {"range": [35, 65], "color": "rgba(217, 119, 6, 0.10)"},
                    {"range": [65, 100], "color": "rgba(220, 38, 38, 0.10)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=18, r=18, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#111827"},
    )
    return fig


def build_zone_map(latest_score: float) -> pd.DataFrame:
    zones = RENNES_ZONES.copy()
    zones["score"] = (latest_score * zones["weight"]).clip(0, 100).round(1)
    zones["niveau"] = zones["score"].apply(score_label)
    zones["taille"] = 18 + zones["score"] / 3
    return zones


def model_predict(input_data: dict, fallback_model) -> float:
    row = pd.DataFrame([input_data])[FEATURES]
    if fallback_model is not None:
        return round(float(fallback_model.predict(row)[0]), 2)
    model, *_ = train(get_data())
    return round(float(model.predict(row)[0]), 2)


def build_decision_summary(
    status: str,
    score: float,
    pollution_score: float,
    traffic_score: float,
    traffic_source: str,
    latest_row: pd.Series = None,
) -> str:
    dominant = "pollution" if pollution_score >= traffic_score else "trafic"

    context_parts = []
    if latest_row is not None:
        no2 = latest_row.get("nitrogen_dioxide", 0)
        hour = latest_row.get("hour", -1)
        is_rush = hour in [7, 8, 9, 17, 18, 19]
        is_rainy = latest_row.get("rain_flag", 0) == 1
        if no2 > 40:
            context_parts.append(f"Le NO2 ({no2:.1f} ug/m3) depasse le seuil OMS de 40 ug/m3")
        if is_rush:
            context_parts.append("Periode d'heure de pointe detectee")
        if is_rainy:
            context_parts.append("Pluie en cours (dispersion naturelle du NO2)")

    context = ". ".join(context_parts) + "." if context_parts else ""

    return (
        f"Situation {status.lower()} avec un EcoTraffic Score de {score:.0f}/100. "
        f"Le facteur dominant est la {dominant} ({pollution_score:.0f} vs {traffic_score:.0f}). "
        f"{context}"
    )


def decision_actions(status: str, latest_row: pd.Series = None) -> list[dict]:
    hour = int(latest_row.get("hour", 12)) if latest_row is not None else 12
    no2 = float(latest_row.get("nitrogen_dioxide", 0)) if latest_row is not None else 0
    is_rush = hour in [7, 8, 9, 17, 18, 19]
    wind = float(latest_row.get("wind_speed_10m", 5)) if latest_row is not None else 5

    if status == "Critique":
        actions = [
            {
                "titre": "Reduction du trafic",
                "detail": "Activer la circulation alternee ou limiter l'acces "
                "aux vehicules les plus polluants (Crit'Air 4-5) dans le centre de Rennes.",
            },
            {
                "titre": "Vitesse reduite",
                "detail": "Abaisser la vitesse a 70 km/h sur la rocade et 30 km/h en centre-ville "
                "pour reduire les emissions de NO2 liees a l'acceleration.",
            },
            {
                "titre": "Alerte population",
                "detail": f"NO2 a {no2:.1f} ug/m3 — informer les populations sensibles "
                "(enfants, asthmatiques) de limiter les activites en exterieur.",
            },
            {
                "titre": "Transports alternatifs",
                "detail": "Renforcer la frequence des bus/metro et rendre le stationnement relais gratuit "
                "pour inciter au report modal.",
            },
        ]
    elif status == "Vigilance":
        actions = [
            {
                "titre": "Optimisation des feux",
                "detail": "Activer les ondes vertes sur les axes principaux pour fluidifier le trafic "
                "et reduire les arrets/redemarrages (pics de NO2).",
            },
            {
                "titre": "Zones 30 temporaires",
                "detail": "Etendre les zones 30 aux quartiers residentiels proches des axes charges "
                "pendant les heures de pointe." if is_rush else
                "Maintenir les zones 30 existantes et surveiller l'evolution.",
            },
            {
                "titre": "Incitation mobilite douce",
                "detail": "Communiquer sur les itineraires velo et les transports en commun "
                "pour les trajets domicile-travail aux heures de pointe.",
            },
        ]
    else:
        actions = [
            {
                "titre": "Prevention",
                "detail": "Conditions favorables — profiter de cette periode pour planifier "
                "des actions structurelles (pistes cyclables, bornes de recharge).",
            },
            {
                "titre": "Analyse des tendances",
                "detail": "Comparer les niveaux actuels avec l'historique pour identifier "
                "les heures et jours ou le NO2 augmente regulierement.",
            },
            {
                "titre": "Ventilation naturelle",
                "detail": f"Vent a {wind:.1f} m/s — conditions {'favorables' if wind > 3 else 'faibles'} "
                "pour la dispersion des polluants.",
            },
        ]

    return actions


def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


traffic_df = get_traffic_data()
df = add_scores(get_data(), traffic_df)
model, metrics, importance, X_test, y_test = get_model_outputs(df)
saved_model = get_saved_model()

_now = datetime.now()
next_refresh_str = (_now + timedelta(seconds=3600 - (_now.minute * 60 + _now.second) % 3600)).strftime("%H:%M")
loaded_at_str = _now.strftime("%d/%m/%Y %H:%M")

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">ET</div>
            <span class="sidebar-logo-name">EcoTraffic</span>
        </div>
        <span class="sidebar-section-label">Menu</span>
        """,
        unsafe_allow_html=True,
    )
    min_date = df["datetime"].min()
    max_date = df["datetime"].max()
    selected_range = st.date_input(
        "Periode",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
        filtered_df = df[
            (df["datetime"].dt.date >= start_date)
            & (df["datetime"].dt.date <= end_date)
        ]
    else:
        filtered_df = df

    filtered_traffic_df = traffic_df

    pollutant = "nitrogen_dioxide"
    smooth_window = 3
    traffic_status_filter = []
    if not traffic_df.empty:
        traffic_statuses = sorted(
            traffic_df["Statut Trafic"].astype(str).str.lower().dropna().unique().tolist()
        )
        traffic_status_filter = st.multiselect(
            "Statuts trafic",
            options=traffic_statuses,
            default=traffic_statuses,
        )
    st.markdown('<span class="sidebar-section-label">Donnees</span>', unsafe_allow_html=True)
    if st.button("Rafraichir le trafic", use_container_width=True):
        with st.spinner("Recuperation des donnees trafic Rennes..."):
            result = subprocess.run(
                [sys.executable, str(ROOT_DIR / "scripts" / "build_traffic_cleaned.py")],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=120,
            )
        if result.returncode == 0:
            get_traffic_data.clear()
            st.success("Donnees trafic mises a jour.")
            st.rerun()
        else:
            st.error("La mise a jour trafic a echoue.")
            st.code(result.stderr or result.stdout)

    if traffic_df.empty:
        st.caption("Trafic non charge. Lance `python scripts/build_traffic_cleaned.py` pour activer les donnees trafic reelles.")
    else:
        st.caption(
            f"Trafic charge: {len(traffic_df)} lignes. "
            f"Periode trafic: {traffic_df['Date/Heure'].min().strftime('%d/%m/%Y %H:%M')}."
        )
    st.markdown(
        f"""
        <div class="sidebar-promo">
            <div class="sidebar-promo-title">Pipeline Airflow</div>
            <div class="sidebar-promo-text">
                Collecte automatique toutes les heures.<br>
                Prochain refresh auto : <strong style="color:#FFFFFF">{next_refresh_str}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if filtered_df.empty:
    st.warning("Aucune donnee disponible pour cette periode.")
    st.stop()

latest = filtered_df.sort_values("datetime").iloc[-1]
score = float(latest["ecotraffic_score"])
pollution_score = float(latest["pollution_score"])
traffic_score = float(latest["traffic_score"])
traffic_source = latest.get("traffic_source", "proxy")
status = score_label(score)
env_last_update = filtered_df["datetime"].max().strftime("%d/%m/%Y %H:%M")
traffic_last_update = (
    traffic_df["Date/Heure"].max().strftime("%d/%m/%Y %H:%M")
    if not traffic_df.empty
    else "non disponible"
)

_status_cls = {"Normal": "status-normal", "Vigilance": "status-vigilance", "Critique": "status-critique"}.get(status, "status-normal")

st.markdown(
    f"""
    <div class="page-header">
        <div>
            <h1 class="page-title">Dashboard</h1>
            <p class="page-subtitle">Qualite de l'air, trafic et prediction NO2 — Rennes Metropole</p>
        </div>
        <div class="header-badges">
            <span class="hbadge">Env: {env_last_update}</span>
            <span class="hbadge">Trafic: {traffic_last_update}</span>
            <span class="hbadge green">Refresh: {next_refresh_str}</span>
            <span class="hbadge {_status_cls}">{status} &nbsp;{score:.0f}/100</span>
        </div>
    </div>
    <div class="kpi-row">
        <div class="kpi-card featured">
            <div class="kpi-top">
                <span class="kpi-label">EcoTraffic Score</span>
                <div class="kpi-arrow">&#8599;</div>
            </div>
            <div class="kpi-value">{score:.0f}<small>/100</small></div>
            <span class="kpi-badge">{status}</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">NO2 actuel</span>
                <div class="kpi-arrow">&#8599;</div>
            </div>
            <div class="kpi-value">{latest[TARGET]:.1f}<small> µg/m³</small></div>
            <span class="kpi-badge">Dioxyde azote</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Score trafic</span>
                <div class="kpi-arrow">&#8599;</div>
            </div>
            <div class="kpi-value">{traffic_score:.0f}<small>/100</small></div>
            <span class="kpi-badge">{str(traffic_source)}</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">PM2.5</span>
                <div class="kpi-arrow">&#8599;</div>
            </div>
            <div class="kpi-value">{latest['pm2_5']:.1f}<small> µg/m³</small></div>
            <span class="kpi-badge">Particules fines</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <span class="kpi-label">Temperature</span>
                <div class="kpi-arrow">&#8599;</div>
            </div>
            <div class="kpi-value">{latest['temperature_2m']:.1f}<small>°C</small></div>
            <span class="kpi-badge">Meteo</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not traffic_df.empty:
    env_date = filtered_df["datetime"].max().date()
    traffic_date = traffic_df["Date/Heure"].max().date()
    if env_date != traffic_date:
        st.info(
            f"Les donnees environnement datent du {env_date.strftime('%d/%m/%Y')}, "
            f"les donnees trafic du {traffic_date.strftime('%d/%m/%Y')}. "
            "Le dashboard utilise le trafic reel le plus recent pour le score global."
        )

tab_control, tab_traffic, tab_map, tab_model, tab_predict, tab_report, tab_data = st.tabs(
    [
        "Vue globale",
        "Trafic",
        "Carte Rennes",
        "Modele ML",
        "Prediction NO2",
        "Actions anti-pollution",
        "Qualite des donnees",
    ]
)

with tab_control:
    cc_score_col, cc_decomp_col = st.columns([1, 1])

    with cc_score_col:
        st.subheader("Score global EcoTraffic")
        st.plotly_chart(gauge("Score EcoTraffic", score), use_container_width=True)
        st.markdown(
            f"""
            <div class="small-card">
                <strong>Comment est calcule ce score ?</strong>
                <span>
                    Le score combine deux composantes :<br>
                    &bull; <b>Pollution (62%)</b> : moyenne ponderee de NO2 (38%), PM10 (24%), PM2.5 (24%) et CO (14%)<br>
                    &bull; <b>Trafic (38%)</b> : base sur le statut des routes, la vitesse moyenne et le temps de trajet<br><br>
                    <b>Score actuel : {score:.0f}/100</b> &rarr; Statut <b>{status}</b>
                    (0-35 = Normal, 35-65 = Vigilance, 65-100 = Critique)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cc_decomp_col:
        st.subheader("Decomposition du score")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.plotly_chart(gauge("Pollution", pollution_score), use_container_width=True)
        with dc2:
            st.plotly_chart(gauge("Trafic", traffic_score), use_container_width=True)
        dominant = "pollution" if pollution_score >= traffic_score else "trafic"
        st.markdown(
            f"""
            <div class="small-card">
                <strong>Analyse de la situation</strong>
                <span>
                    Le facteur dominant est la <b>{dominant}</b>
                    (pollution {pollution_score:.0f}/100 vs trafic {traffic_score:.0f}/100).<br>
                    Source trafic : <b>{str(traffic_source)}</b>.<br><br>
                    &bull; <b>Score trafic reel</b> : 50% statut route (freeflow/heavy/congested)
                    + 35% pression vitesse (1 - moy/max) + 15% temps de trajet<br>
                    &bull; <b>Score proxy</b> (si pas de donnees trafic) : heures de pointe, pluie, CO, vent
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Evolution de la pollution")
    trend = filtered_df.copy()
    trend["pollutant_smoothed"] = trend[pollutant].rolling(smooth_window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["datetime"],
            y=trend[pollutant],
            mode="lines",
            name="Mesure brute",
            line=dict(color="rgba(37, 99, 235, 0.35)", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend["datetime"],
            y=trend["pollutant_smoothed"],
            mode="lines",
            name=f"Tendance (lissage {smooth_window}h)",
            line=dict(color="#0CAF6D", width=3),
        )
    )
    fig.update_layout(
        title=f"{pollutant} — mesure horaire et tendance lissee",
        height=360,
        margin=dict(l=12, r=12, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
        legend=dict(orientation="h"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)", title="ug/m3"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparaison des polluants")
    pollution_long = filtered_df.melt(
        id_vars=["datetime"],
        value_vars=POLLUTANTS,
        var_name="polluant",
        value_name="valeur",
    )
    fig_pollutants = px.area(
        pollution_long,
        x="datetime",
        y="valeur",
        color="polluant",
        line_group="polluant",
        labels={"datetime": "Date", "valeur": "Valeur", "polluant": "Polluant"},
    )
    fig_pollutants.update_layout(
        height=390,
        margin=dict(l=12, r=12, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111827"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
    )
    st.plotly_chart(fig_pollutants, use_container_width=True)

with tab_traffic:
    st.subheader("Analyse trafic Rennes")
    if filtered_traffic_df.empty:
        st.warning(
            "Aucun fichier trafic nettoye n'est charge. Lance `python scripts/build_traffic_cleaned.py`, puis recharge le dashboard."
        )
        st.code("python scripts/build_traffic_cleaned.py", language="bash")
    else:
        traffic_view = filtered_traffic_df.copy()
        if traffic_status_filter:
            traffic_view = traffic_view[
                traffic_view["Statut Trafic"].astype(str).str.lower().isin(traffic_status_filter)
            ]
        if traffic_view.empty:
            st.warning("Aucune donnee trafic ne correspond aux statuts selectionnes.")
            st.stop()

        traffic_view["Vitesse Moyenne (km/h)"] = pd.to_numeric(
            traffic_view["Vitesse Moyenne (km/h)"], errors="coerce"
        )
        traffic_view["Vitesse Max (km/h)"] = pd.to_numeric(
            traffic_view["Vitesse Max (km/h)"], errors="coerce"
        )
        traffic_view["Temps de Trajet (s)"] = pd.to_numeric(
            traffic_view["Temps de Trajet (s)"], errors="coerce"
        )
        traffic_view["datetime_hour"] = traffic_view["Date/Heure"].dt.floor("h")
        traffic_last = traffic_view["Date/Heure"].max().strftime("%d/%m/%Y %H:%M")

        route_count = traffic_view["Route"].nunique()
        avg_speed_value = traffic_view["Vitesse Moyenne (km/h)"].mean()
        slow_routes = (
            traffic_view["Vitesse Moyenne (km/h)"]
            < traffic_view["Vitesse Max (km/h)"] * 0.45
        ).sum()
        congested_count = traffic_view["Statut Trafic"].astype(str).str.lower().eq("congested").sum()

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Routes observees", f"{route_count}")
        t2.metric("Vitesse moyenne", f"{avg_speed_value:.1f} km/h")
        t3.metric("Routes lentes", f"{int(slow_routes)}")
        t4.metric("Congestion", f"{int(congested_count)}")
        t5.metric("Maj trafic", traffic_last)

        hourly_traffic = compute_traffic_score(traffic_view)
        
        if not hourly_traffic.empty:
            st.plotly_chart(
                gauge("Score trafic reel", float(hourly_traffic["traffic_real_score"].iloc[-1])),
                use_container_width=True,
            )

        status_counts = (
            traffic_view["Statut Trafic"]
            .astype(str)
            .str.lower()
            .value_counts()
            .reset_index()
        )
        status_counts.columns = ["Statut", "Nombre"]

        route_speed = (
            traffic_view.groupby("Route", as_index=False)
            .agg(
                vitesse_moyenne=("Vitesse Moyenne (km/h)", "mean"),
                vitesse_max=("Vitesse Max (km/h)", "mean"),
                observations=("Route", "size"),
            )
            .sort_values("vitesse_moyenne")
            .head(15)
        )

        c_status, c_routes = st.columns([0.8, 1.2])
        with c_status:
            fig_status = px.pie(
                status_counts,
                names="Statut",
                values="Nombre",
                hole=0.48,
                color="Statut",
                color_discrete_map={
                    "freeflow": "#0f9f6e",
                    "heavy": "#c77700",
                    "congested": "#c93636",
                    "unknown": "#6b7280",
                },
            )
            fig_status.update_layout(
                height=390,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#111827"),
            )
            st.plotly_chart(fig_status, use_container_width=True)

        with c_routes:
            fig_routes = px.bar(
                route_speed,
                x="vitesse_moyenne",
                y="Route",
                orientation="h",
                color="vitesse_moyenne",
                color_continuous_scale=["#c93636", "#c77700", "#0f9f6e"],
                labels={
                    "vitesse_moyenne": "Vitesse moyenne (km/h)",
                    "Route": "Route",
                },
            )
            fig_routes.update_layout(
                height=390,
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=8, r=8, t=12, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#111827"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_routes, use_container_width=True)

        st.dataframe(traffic_view.tail(30), use_container_width=True, hide_index=True)

with tab_map:
    st.subheader("Carte operationnelle de Rennes")

    STATUS_COLORS = {
        "freeflow": "#0CAF6D",
        "freeFlow": "#0CAF6D",
        "heavy": "#D97706",
        "congested": "#DC2626",
        "unknown": "#9CA3AF",
    }

    has_traffic_coordinates = (
        not traffic_df.empty
        and {"Latitude", "Longitude"}.issubset(traffic_df.columns)
        and traffic_df[["Latitude", "Longitude"]].notna().all(axis=1).any()
    )

    if has_traffic_coordinates:
        map_traffic = traffic_df.dropna(subset=["Latitude", "Longitude"]).copy()
        if traffic_status_filter:
            map_traffic = map_traffic[
                map_traffic["Statut Trafic"].astype(str).str.lower().isin(traffic_status_filter)
            ]
        map_traffic["Statut Trafic"] = map_traffic["Statut Trafic"].astype(str).str.lower()
        map_traffic["Vitesse Moyenne (km/h)"] = pd.to_numeric(
            map_traffic["Vitesse Moyenne (km/h)"], errors="coerce"
        )
        st.markdown(
            '<p class="section-note">Carte basee sur les points GPS reels des routes — couleurs par statut trafic.</p>',
            unsafe_allow_html=True,
        )
        fig_map = px.scatter_mapbox(
            map_traffic,
            lat="Latitude",
            lon="Longitude",
            color="Statut Trafic",
            size="Vitesse Moyenne (km/h)",
            hover_name="Route",
            hover_data={
                "Vitesse Moyenne (km/h)": True,
                "Vitesse Max (km/h)": True,
                "Statut Trafic": True,
                "Latitude": False,
                "Longitude": False,
            },
            color_discrete_map=STATUS_COLORS,
            zoom=11,
            height=560,
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                title="Statut trafic",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.10)",
                borderwidth=1,
                font=dict(color="#111827"),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    else:
        # Fallback: zone map
        if not traffic_df.empty:
            st.info(
                "Coordonnees GPS absentes du fichier traffic_cleaned.csv. "
                "Cliquez sur **Rafraichir le trafic** dans la barre laterale pour activer "
                "la carte par route avec les statuts freeflow / heavy / congested."
            )
        zones = build_zone_map(score)
        fig_map = px.scatter_mapbox(
            zones,
            lat="lat",
            lon="lon",
            color="niveau",
            size="taille",
            hover_name="zone",
            hover_data={"score": True, "lat": False, "lon": False, "taille": False},
            color_discrete_map={
                "Normal": "#0CAF6D",
                "Vigilance": "#D97706",
                "Critique": "#DC2626",
            },
            zoom=11,
            height=480,
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                title="Niveau EcoTraffic",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.10)",
                borderwidth=1,
                font=dict(color="#111827"),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Affiche la répartition des statuts trafic même sans GPS
        if not traffic_df.empty:
            st.subheader("Repartition des statuts trafic")
            st.markdown(
                '<p class="section-note">Distribution des 89 routes par statut — donnees disponibles meme sans coordonnees GPS.</p>',
                unsafe_allow_html=True,
            )
            status_counts = (
                traffic_df["Statut Trafic"]
                .astype(str).str.lower().str.strip()
                .value_counts()
                .reset_index()
            )
            status_counts.columns = ["Statut", "Nombre"]
            status_counts["Couleur"] = status_counts["Statut"].map(STATUS_COLORS).fillna("#9CA3AF")

            col_bar, col_donut = st.columns([1.4, 1])
            with col_bar:
                fig_status_bar = px.bar(
                    status_counts,
                    x="Statut",
                    y="Nombre",
                    color="Statut",
                    color_discrete_map=STATUS_COLORS,
                    text="Nombre",
                    labels={"Statut": "Statut trafic", "Nombre": "Nombre de routes"},
                )
                fig_status_bar.update_traces(textposition="outside")
                fig_status_bar.update_layout(
                    height=320,
                    margin=dict(l=8, r=8, t=12, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#111827"),
                    showlegend=False,
                    xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                )
                st.plotly_chart(fig_status_bar, use_container_width=True)

            with col_donut:
                fig_status_pie = px.pie(
                    status_counts,
                    names="Statut",
                    values="Nombre",
                    hole=0.50,
                    color="Statut",
                    color_discrete_map=STATUS_COLORS,
                )
                fig_status_pie.update_layout(
                    height=320,
                    margin=dict(l=8, r=8, t=12, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#111827"),
                    legend=dict(font=dict(color="#111827")),
                )
                st.plotly_chart(fig_status_pie, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="small-card"><strong>Couche actuelle</strong><span>Meteo + pollution pretraitees depuis le pipeline Kedro.</span></div>', unsafe_allow_html=True)
    if traffic_df.empty:
        c2.markdown('<div class="small-card"><strong>Couche trafic</strong><span>Lance build_traffic_cleaned.py ou clique Rafraichir pour activer routes, vitesses et statuts.</span></div>', unsafe_allow_html=True)
    elif has_traffic_coordinates:
        c2.markdown('<div class="small-card"><strong>Couche trafic</strong><span>Routes GPS actives — freeflow / heavy / congested visibles sur la carte.</span></div>', unsafe_allow_html=True)
    else:
        c2.markdown('<div class="small-card"><strong>Couche trafic</strong><span>Donnees trafic chargees sans GPS. Rafraichir pour activer la carte par route.</span></div>', unsafe_allow_html=True)

with tab_model:
    st.subheader("Comparaison de 6 modeles ML")

    @st.cache_data(ttl=3600)
    def load_comparison_data():
        if not MODELS_COMPARISON_PATH.exists():
            return None
        with open(MODELS_COMPARISON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    comparison_data = load_comparison_data()

    if comparison_data is not None:
        valid_models = {k: v for k, v in comparison_data.items() if v is not None}

        if valid_models:
            comp_df = pd.DataFrame([
                {
                    "Modele": name,
                    "Rang": data["rank"],
                    "MAE": data["metrics"]["MAE"],
                    "RMSE": data["metrics"]["RMSE"],
                    "R2": data["metrics"]["R2"],
                    "CV Mean R2": data["metrics"].get("CV_Mean"),
                    "CV Std": data["metrics"].get("CV_Std"),
                    "Temps (s)": data["metrics"].get("training_time_sec"),
                    "Justification": data["justification"],
                }
                for name, data in valid_models.items()
            ]).sort_values("Rang")

            best = comp_df.iloc[0]

            st.markdown(
                f"""
                <div class="small-card" style="border-left: 4px solid var(--green);">
                    <strong>Modele gagnant : {best['Modele']}</strong>
                    <span>
                        R2 = {best['R2']:.3f} | MAE = {best['MAE']:.3f} | RMSE = {best['RMSE']:.3f}<br>
                        {best['Justification']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            mc1, mc2 = st.columns(2)

            with mc1:
                fig_r2 = px.bar(
                    comp_df,
                    x="R2",
                    y="Modele",
                    orientation="h",
                    color="R2",
                    color_continuous_scale=["#DC2626", "#D97706", "#0CAF6D"],
                    text="R2",
                )
                fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_r2.update_layout(
                    title="R2 par modele (plus haut = meilleur)",
                    height=380,
                    yaxis=dict(categoryorder="total ascending", gridcolor="rgba(0,0,0,0.06)"),
                    margin=dict(l=8, r=60, t=40, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#111827"),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                )
                st.plotly_chart(fig_r2, use_container_width=True)

            with mc2:
                fig_mae = px.bar(
                    comp_df,
                    x="MAE",
                    y="Modele",
                    orientation="h",
                    color="MAE",
                    color_continuous_scale=["#0CAF6D", "#D97706", "#DC2626"],
                    text="MAE",
                )
                fig_mae.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_mae.update_layout(
                    title="MAE par modele (plus bas = meilleur)",
                    height=380,
                    yaxis=dict(categoryorder="total descending", gridcolor="rgba(0,0,0,0.06)"),
                    margin=dict(l=8, r=60, t=40, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#111827"),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                )
                st.plotly_chart(fig_mae, use_container_width=True)

            mc3, mc4 = st.columns(2)

            with mc3:
                fig_radar = go.Figure()
                for _, row in comp_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row["R2"], 1 - row["MAE"] / comp_df["MAE"].max(),
                           1 - row["RMSE"] / comp_df["RMSE"].max(),
                           row["CV Mean R2"] if row["CV Mean R2"] and row["CV Mean R2"] > 0 else 0,
                           1 - (row["Temps (s)"] / comp_df["Temps (s)"].max()) if row["Temps (s)"] else 0],
                        theta=["R2", "1 - MAE norm", "1 - RMSE norm", "CV R2", "Rapidite"],
                        fill="toself",
                        name=row["Modele"],
                        opacity=0.6,
                    ))
                fig_radar.update_layout(
                    title="Profil multi-criteres (plus grand = meilleur)",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=420,
                    margin=dict(l=40, r=40, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#111827"),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with mc4:
                fig_cv = go.Figure()
                for _, row in comp_df.iterrows():
                    cv_mean = row["CV Mean R2"] if row["CV Mean R2"] is not None else 0
                    cv_std = row["CV Std"] if row["CV Std"] is not None else 0
                    fig_cv.add_trace(go.Bar(
                        x=[row["Modele"]],
                        y=[cv_mean],
                        error_y=dict(type="data", array=[cv_std], visible=True),
                        name=row["Modele"],
                        showlegend=False,
                    ))
                fig_cv.update_layout(
                    title="Cross-validation R2 (5-fold) avec ecart-type",
                    height=420,
                    margin=dict(l=8, r=8, t=50, b=8),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#111827"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0.06)", title="R2 moyen"),
                )
                st.plotly_chart(fig_cv, use_container_width=True)

            st.subheader("Tableau comparatif complet")
            display_df = comp_df[["Rang", "Modele", "R2", "MAE", "RMSE", "CV Mean R2", "CV Std", "Temps (s)", "Justification"]]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Analyse et conclusion")

            analysis_col1, analysis_col2 = st.columns(2)
            with analysis_col1:
                st.markdown(
                    f"""
                    <div class="small-card">
                        <strong>Pourquoi {best['Modele']} gagne</strong>
                        <span>
                            Meilleur R2 ({best['R2']:.3f}) et meilleur MAE ({best['MAE']:.3f}).
                            {best['Justification']}
                            Sur un dataset de 109 echantillons avec des features heterogenes
                            (meteo + pollution), le gradient boosting capture les relations
                            non-lineaires mieux que les modeles lineaires ou les reseaux de neurones.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with analysis_col2:
                worst = comp_df.iloc[-1]
                st.markdown(
                    f"""
                    <div class="small-card">
                        <strong>Pourquoi {worst['Modele']} est dernier</strong>
                        <span>
                            R2 = {worst['R2']:.3f} seulement. {worst['Justification']}
                            Avec seulement 109 echantillons, les reseaux de neurones et modeles
                            lineaires manquent de donnees ou de capacite pour capturer la complexite
                            des interactions pollution-meteo.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Aucun modele valide dans le fichier de comparaison.")
    elif comparison_data is None:
        st.info(
            "La comparaison de modeles n'a pas encore ete executee. "
            "Lance `python ml/compare_models.py` pour generer les resultats."
        )

    st.markdown("---")
    st.subheader("Modele XGBoost - Prediction NO2")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", metrics["MAE"])
    m2.metric("RMSE", metrics["RMSE"])
    m3.metric("R2", metrics["R2"])
    m4.metric("Algorithme", "XGBoost")

    y_pred = model.predict(X_test)
    errors = y_pred - y_test.values

    viz_top1, viz_top2 = st.columns(2)

    with viz_top1:
        lim_min = min(float(y_test.min()), float(y_pred.min())) - 0.5
        lim_max = max(float(y_test.max()), float(y_pred.max())) + 0.5
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(
            x=y_test.values, y=y_pred, mode="markers",
            marker=dict(color="#2196F3", size=10, line=dict(color="white", width=1)),
            name="Predictions",
        ))
        fig_rv.add_trace(go.Scatter(
            x=[lim_min, lim_max], y=[lim_min, lim_max], mode="lines",
            line=dict(color="red", dash="dash", width=1.5),
            name="Prediction parfaite",
        ))
        fig_rv.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper", showarrow=False,
            text=f"R2 = {metrics['R2']}<br>MAE = {metrics['MAE']} ug/m3",
            bgcolor="lightyellow", bordercolor="#ccc", borderwidth=1,
            font=dict(size=12),
        )
        fig_rv.update_layout(
            title="Reel vs Predit",
            xaxis_title="NO2 reel (ug/m3)",
            yaxis_title="NO2 predit (ug/m3)",
            height=380,
            margin=dict(l=12, r=12, t=40, b=12),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
            font=dict(color="#111827"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )
        st.plotly_chart(fig_rv, use_container_width=True)

    with viz_top2:
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=list(range(len(errors))), y=errors,
            marker_color=["#F44336" if e > 0 else "#4CAF50" for e in errors],
            opacity=0.8,
        ))
        fig_err.add_hline(y=0, line_color="black", line_width=1)
        fig_err.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper", showarrow=False,
            text=f"RMSE = {metrics['RMSE']} ug/m3",
            bgcolor="lightyellow", bordercolor="#ccc", borderwidth=1,
            font=dict(size=12),
        )
        fig_err.update_layout(
            title="Erreurs de prediction",
            xaxis_title="Echantillon de test",
            yaxis_title="Erreur (predit - reel) ug/m3",
            height=380, showlegend=False,
            margin=dict(l=12, r=12, t=40, b=12),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
            font=dict(color="#111827"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    viz_bot1, viz_bot2 = st.columns(2)

    with viz_bot1:
        fig_fi = px.bar(
            importance, x=importance["importance"] * 100, y="feature",
            orientation="h", text=importance["importance"].apply(lambda v: f"{v*100:.1f}%"),
            color=importance["importance"] * 100,
            color_continuous_scale=["#90CAF9", "#1976D2"],
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            title="Importance des features",
            xaxis_title="Importance (%)", yaxis_title="",
            height=380, yaxis=dict(autorange="reversed"),
            margin=dict(l=12, r=60, t=40, b=12),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
            font=dict(color="#111827"), coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with viz_bot2:
        heures = list(range(24))
        base_scenario = {
            "temperature_2m": 15.0, "precipitation": 0.0,
            "wind_speed_10m": 5.0, "wind_kmh": 18.0,
            "rain_flag": 0, "carbon_monoxide": 140.0,
            "ozone": 65.0, "day_of_week": 0,
            "is_weekend": 0, "month": 5,
        }
        scenarios = {
            "Lundi (semaine)": {**base_scenario, "day_of_week": 0, "is_weekend": 0},
            "Samedi (weekend)": {**base_scenario, "day_of_week": 5, "is_weekend": 1},
            "Jour de pluie": {**base_scenario, "precipitation": 1.5, "rain_flag": 1},
        }
        sc_colors = ["#1976D2", "#FF9800", "#4CAF50"]
        fig_sc = go.Figure()
        for (label, params), color in zip(scenarios.items(), sc_colors):
            preds_sc = [float(model.predict(pd.DataFrame([{**params, "hour": h}])[FEATURES])[0]) for h in heures]
            fig_sc.add_trace(go.Scatter(
                x=heures, y=preds_sc, mode="lines+markers",
                name=label, line=dict(color=color, width=2),
                marker=dict(size=5),
            ))
        fig_sc.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Pointe matin", annotation_position="top left")
        fig_sc.add_vrect(x0=17, x1=19, fillcolor="orange", opacity=0.08, line_width=0, annotation_text="Pointe soir", annotation_position="top left")
        fig_sc.update_layout(
            title="NO2 predit selon l'heure - 3 scenarios",
            xaxis_title="Heure de la journee",
            yaxis_title="NO2 predit (ug/m3)",
            height=380,
            margin=dict(l=12, r=12, t=40, b=12),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
            font=dict(color="#111827"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)", dtick=2),
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

with tab_predict:
    st.subheader("Simulation de scenario NO2")
    st.markdown(
        '<p class="section-note">Modifie les conditions meteo et pollution pour estimer la concentration de NO2 predite par le modele.</p>',
        unsafe_allow_html=True,
    )

    form_col1, form_col2, form_col3 = st.columns(3)
    with form_col1:
        temperature = st.slider("Temperature (C)", -10.0, 40.0, get_default_value(df, "temperature_2m", 15.0), 0.1)
        precipitation = st.slider("Precipitation (mm)", 0.0, 20.0, get_default_value(df, "precipitation", 0.0), 0.1)
        wind_speed = st.slider("Vent (m/s)", 0.0, 30.0, get_default_value(df, "wind_speed_10m", 5.0), 0.1)
        wind_kmh = st.slider("Vent (km/h)", 0.0, 120.0, get_default_value(df, "wind_kmh", 18.0), 0.1)

    with form_col2:
        carbon_monoxide = st.slider("Monoxyde de carbone", 0.0, 500.0, get_default_value(df, "carbon_monoxide", 140.0), 1.0)
        ozone = st.slider("Ozone", 0.0, 200.0, get_default_value(df, "ozone", 65.0), 1.0)
        hour = st.slider("Heure", 0, 23, int(get_default_value(df, "hour", 8)))
        month = st.slider("Mois", 1, 12, int(get_default_value(df, "month", 5)))

    with form_col3:
        day_of_week = st.selectbox(
            "Jour",
            options=list(range(7)),
            format_func=lambda d: ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"][d],
        )
        rain_flag = st.toggle("Pluie", value=bool(precipitation > 0))
        is_weekend = day_of_week >= 5

    input_data = {
        "temperature_2m": temperature,
        "precipitation": precipitation,
        "wind_speed_10m": wind_speed,
        "wind_kmh": wind_kmh,
        "rain_flag": int(rain_flag),
        "carbon_monoxide": carbon_monoxide,
        "ozone": ozone,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": int(is_weekend),
        "month": month,
    }

    no2_prediction = model_predict(input_data, saved_model or model)
    p1, p2 = st.columns([0.8, 1.2])
    with p1:
        st.metric("NO2 predit", f"{no2_prediction:.2f} ug/m3")
        projected_score = min(100, (no2_prediction / max(df[TARGET].max(), 1)) * 100)
        st.plotly_chart(gauge("Risque NO2", projected_score), use_container_width=True)
    with p2:
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True, hide_index=True)

with tab_report:
    st.subheader("Rapport decisionnel")
    st.markdown(
        '<p class="section-note">'
        "Ce rapport propose des actions concretes pour reduire le NO2 a Rennes, "
        "adaptees en temps reel au niveau de risque (Normal, Vigilance, Critique). "
        "Les recommandations s'appuient sur le score EcoTraffic, les donnees meteo, "
        "le niveau de pollution et les conditions de trafic."
        "</p>",
        unsafe_allow_html=True,
    )

    summary = build_decision_summary(
        status=status,
        score=score,
        pollution_score=pollution_score,
        traffic_score=traffic_score,
        traffic_source=str(traffic_source),
        latest_row=latest,
    )
    st.markdown(
        f"""
        <div class="small-card">
            <strong>Synthese automatique</strong>
            <span>{summary}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Statut", status)
    r2.metric("Score global", f"{score:.0f}/100")
    r3.metric("Pollution", f"{pollution_score:.0f}/100")
    r4.metric("Trafic", f"{traffic_score:.0f}/100")

    action_col, context_col = st.columns([1, 1])
    with action_col:
        st.subheader("Actions pour reduire le NO2")
        for action in decision_actions(status, latest_row=latest):
            st.markdown(
                f"""
                <div class="small-card">
                    <strong>{action['titre']}</strong>
                    <span>{action['detail']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with context_col:
        st.subheader("Contexte des donnees")
        st.markdown(
            f"""
            <div class="small-card">
                <strong>Fraicheur</strong>
                <span>Environnement: {env_last_update}<br>Trafic: {traffic_last_update}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="small-card">
                <strong>Points d'attention</strong>
                <span>Les donnees trafic et environnement peuvent couvrir des periodes differentes. Le dashboard l'indique pour garder une lecture transparente.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    export_env, export_traffic = st.columns(2)
    with export_env:
        st.download_button(
            "Telecharger les donnees environnement filtrees",
            data=to_csv_bytes(filtered_df),
            file_name="ecotraffic_environment_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_traffic:
        if filtered_traffic_df.empty:
            st.info("Aucune donnee trafic disponible a exporter.")
        else:
            st.download_button(
                "Telecharger les donnees trafic",
                data=to_csv_bytes(filtered_traffic_df),
                file_name="ecotraffic_traffic_cleaned.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Reference des actions par niveau de risque")
    actions_ref = pd.DataFrame([
        {"Niveau": "Normal (0-35)", "Action": "Prevention", "Detail": "Planifier des actions structurelles (pistes cyclables, bornes de recharge)"},
        {"Niveau": "Normal (0-35)", "Action": "Analyse des tendances", "Detail": "Identifier les heures et jours recurrents de hausse du NO2"},
        {"Niveau": "Normal (0-35)", "Action": "Ventilation naturelle", "Detail": "Evaluer l'impact du vent sur la dispersion des polluants"},
        {"Niveau": "Vigilance (35-65)", "Action": "Optimisation des feux", "Detail": "Ondes vertes sur axes principaux pour reduire arrets/redemarrages (pics NO2)"},
        {"Niveau": "Vigilance (35-65)", "Action": "Zones 30 temporaires", "Detail": "Etendre les zones 30 aux quartiers residentiels pendant les heures de pointe"},
        {"Niveau": "Vigilance (35-65)", "Action": "Incitation mobilite douce", "Detail": "Communication velo et transports en commun pour les trajets domicile-travail"},
        {"Niveau": "Critique (65-100)", "Action": "Reduction du trafic", "Detail": "Circulation alternee, restriction Crit'Air 4-5 dans le centre de Rennes"},
        {"Niveau": "Critique (65-100)", "Action": "Vitesse reduite", "Detail": "70 km/h sur la rocade, 30 km/h en centre-ville"},
        {"Niveau": "Critique (65-100)", "Action": "Alerte population", "Detail": "Informer les populations sensibles (enfants, asthmatiques) de limiter les sorties"},
        {"Niveau": "Critique (65-100)", "Action": "Transports alternatifs", "Detail": "Renforcement bus/metro, parkings relais gratuits pour inciter au report modal"},
    ])
    st.dataframe(actions_ref, use_container_width=True, hide_index=True)

with tab_data:
    st.subheader("Qualite des donnees collectees")

    d1, d2, d3 = st.columns(3)
    d1.metric("Lignes", f"{len(filtered_df)}")
    d2.metric("Debut", filtered_df["datetime"].min().strftime("%d/%m/%Y"))
    d3.metric("Fin", filtered_df["datetime"].max().strftime("%d/%m/%Y"))

    corr_cols = ["temperature_2m", "precipitation", "wind_kmh", "pm10", "pm2_5", "carbon_monoxide", TARGET, "ozone"]
    corr = filtered_df[corr_cols].corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1)
    fig_corr.update_layout(
        title="Correlations entre variables meteo et pollution",
        height=420,
        margin=dict(l=8, r=8, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
