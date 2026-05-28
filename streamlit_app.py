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
        --bg: #F4F6F5;
        --panel: #FFFFFF;
        --green-dark: #1A4731;
        --green-dark-hover: #1F5939;
        --green: #0CAF6D;
        --green-light: #E8F5EF;
        --blue: #2563EB;
        --orange: #D97706;
        --red: #DC2626;
        --text: #111827;
        --text-soft: #6B7280;
        --text-muted: #9CA3AF;
        --border: rgba(0,0,0,0.07);
        --radius: 16px;
        --radius-sm: 10px;
        --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);
        --shadow-md: 0 2px 8px rgba(0,0,0,0.08), 0 8px 24px rgba(0,0,0,0.06);
    }

    * { font-family: 'Inter', sans-serif; box-sizing: border-box; }

    .stApp { background: var(--bg); color: var(--text); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: var(--green-dark) !important;
        border: none !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        transition: background 0.2s;
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--green-dark-hover) !important;
    }

    /* ── Sidebar logo ── */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }
    .sidebar-logo-icon {
        width: 40px; height: 40px;
        background: var(--green-dark);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: 800; color: #FFFFFF;
        flex-shrink: 0;
        letter-spacing: -0.5px;
    }
    .sidebar-logo-name {
        font-size: 18px !important;
        font-weight: 800 !important;
        color: var(--text) !important;
        line-height: 1;
    }
    .sidebar-section-label {
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted) !important;
        margin: 18px 0 8px 2px;
        display: block;
    }
    .sidebar-promo {
        background: var(--green-dark);
        border-radius: 14px;
        padding: 18px;
        margin-top: 24px;
    }
    .sidebar-promo-title {
        font-size: 14px; font-weight: 700; color: #FFFFFF !important;
        margin-bottom: 6px;
    }
    .sidebar-promo-text {
        font-size: 12px; color: rgba(255,255,255,0.70) !important;
        line-height: 1.5;
    }

    /* ── KPI cards ── */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 18px;
        position: relative;
        box-shadow: var(--shadow);
        transition: box-shadow 0.2s, transform 0.15s;
        min-height: 130px;
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .kpi-card:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
    .kpi-card.featured { background: var(--green-dark); border-color: transparent; }
    .kpi-top { display: flex; justify-content: space-between; align-items: flex-start; }
    .kpi-label { font-size: 12px; font-weight: 500; color: var(--text-soft); margin-bottom: 10px; }
    .kpi-card.featured .kpi-label { color: rgba(255,255,255,0.65); }
    .kpi-value {
        font-size: 34px; font-weight: 800; color: var(--text);
        line-height: 1; margin-bottom: 10px;
    }
    .kpi-card.featured .kpi-value { color: #FFFFFF; }
    .kpi-value small { font-size: 14px; font-weight: 500; opacity: 0.55; }
    .kpi-badge {
        display: inline-flex; align-items: center; gap: 4px;
        font-size: 11px; font-weight: 600; padding: 3px 10px;
        border-radius: 20px; background: var(--green-light); color: var(--green);
        max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .kpi-card.featured .kpi-badge { background: rgba(255,255,255,0.15); color: rgba(255,255,255,0.90); }
    .kpi-arrow {
        width: 26px; height: 26px; border-radius: 50%;
        border: 1px solid var(--border);
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; color: var(--text-soft); flex-shrink: 0;
    }
    .kpi-card.featured .kpi-arrow { border-color: rgba(255,255,255,0.22); color: rgba(255,255,255,0.75); }

    /* ── Page header ── */
    .page-header {
        display: flex; justify-content: space-between; align-items: flex-end;
        margin-bottom: 24px; padding-bottom: 18px; border-bottom: 1px solid var(--border);
    }
    .page-title { font-size: 30px !important; font-weight: 800 !important; color: var(--text) !important; margin: 0 !important; }
    .page-subtitle { color: var(--text-soft); font-size: 14px; margin: 4px 0 0 !important; }
    .header-badges { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .hbadge {
        font-size: 12px; border-radius: 8px; padding: 6px 12px; font-weight: 500;
        border: 1px solid var(--border); background: var(--panel); color: var(--text-soft);
    }
    .hbadge.green { background: var(--green-light); color: var(--green); border-color: transparent; font-weight: 600; }
    .hbadge.status-normal  { background: var(--green-light); color: var(--green); border: none; font-weight: 700; }
    .hbadge.status-vigilance { background: #FEF3C7; color: #92400E; border: none; font-weight: 700; }
    .hbadge.status-critique  { background: #FEE2E2; color: #991B1B; border: none; font-weight: 700; }

    /* ── Inner metric cards (tabs) ── */
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 18px 20px;
        box-shadow: var(--shadow);
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--text-soft) !important; font-size: 11px !important;
        font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.7px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text) !important; font-size: 26px !important; font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: var(--green) !important; }
    div[data-testid="stMetricDelta"] svg { fill: var(--green) !important; }

    /* ── Small cards ── */
    .small-card {
        border: 1px solid var(--border); border-radius: var(--radius-sm);
        padding: 16px 18px; background: var(--panel); min-height: 80px;
        box-shadow: var(--shadow); margin-bottom: 10px;
        transition: box-shadow 0.2s, border-color 0.2s;
    }
    .small-card:hover { border-color: rgba(12,175,109,0.25); box-shadow: var(--shadow-md); }
    .small-card strong {
        display: block; font-size: 11px; font-weight: 700; margin-bottom: 6px;
        color: var(--green); text-transform: uppercase; letter-spacing: 0.7px;
    }
    .small-card span { color: var(--text-soft); font-size: 13px; line-height: 1.55; }

    .section-note { color: var(--text-muted); margin-top: -4px; margin-bottom: 14px; font-size: 13px; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; background: transparent !important;
        border-bottom: 2px solid #E5E7EB !important; padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border: none; border-bottom: 2px solid transparent; border-radius: 0;
        background: transparent; padding: 8px 16px;
        color: var(--text-soft) !important; font-size: 13px; font-weight: 500; margin-bottom: -2px;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        border-bottom: 2px solid var(--green-dark) !important;
        color: var(--green-dark) !important; font-weight: 700 !important;
    }

    h1, h2, h3, h4, h5, h6 { color: var(--text); }
    p, span, label { color: var(--text); }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #F1F5F9; }
    ::-webkit-scrollbar-thumb { background: rgba(26,71,49,0.30); border-radius: 3px; }

    div[data-testid="stDownloadButton"] > button {
        background: var(--green-dark) !important; border: none !important;
        color: #FFFFFF !important; border-radius: 10px !important; font-weight: 600 !important;
    }
    div[data-testid="stDownloadButton"] > button:hover { background: var(--green-dark-hover) !important; }

    .stAlert {
        background: #FFFFFF !important; border-color: var(--border) !important;
        color: var(--text) !important; border-radius: var(--radius-sm) !important;
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
    return model, metrics, importance


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
) -> str:
    dominant = "pollution" if pollution_score >= traffic_score else "trafic"
    if status == "Critique":
        action = "surveillance renforcee et analyse prioritaire des zones a risque"
    elif status == "Vigilance":
        action = "suivi regulier de l'evolution pollution/trafic"
    else:
        action = "surveillance standard"

    return (
        f"Situation {status.lower()} avec un EcoTraffic Score de {score:.0f}/100. "
        f"Le facteur dominant est le {dominant}. "
        f"Le score trafic utilise la source: {traffic_source}. "
        f"Action recommandee: {action}."
    )


def decision_actions(status: str) -> list[str]:
    if status == "Critique":
        return [
            "Prioriser la surveillance des axes les plus lents.",
            "Suivre l'evolution du NO2 et des particules fines.",
            "Preparer une communication de vigilance si la situation persiste.",
        ]
    if status == "Vigilance":
        return [
            "Comparer la tendance pollution avec les heures de pointe.",
            "Surveiller les routes avec vitesse moyenne basse.",
            "Rafraichir les donnees trafic avant la prise de decision.",
        ]
    return [
        "Maintenir une surveillance standard.",
        "Verifier la fraicheur des donnees avant diffusion.",
        "Conserver l'historique pour analyser les tendances.",
    ]


def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


traffic_df = get_traffic_data()
df = add_scores(get_data(), traffic_df)
model, metrics, importance = get_model_outputs(df)
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

    pollutant = st.selectbox("Polluant", POLLUTANTS)
    smooth_window = st.slider("Lissage horaire", 1, 12, 3)
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

tab_control, tab_traffic, tab_map, tab_model, tab_predict, tab_report, tab_mvp, tab_data, tab_arch = st.tabs(
    [
        "Command Center",
        "Trafic",
        "Carte Rennes",
        "Modele ML",
        "Simulation",
        "Rapport",
        "MVP",
        "Donnees",
        "Architecture",
    ]
)

with tab_control:
    left, middle, right = st.columns([1.05, 1.15, 0.9])

    with left:
        st.plotly_chart(gauge("Score EcoTraffic", score), use_container_width=True)
        st.markdown(
            f"""
            <div class="small-card">
                <strong>Lecture du score</strong>
                <span>Statut actuel: {status}. 0-35 normal, 35-65 vigilance, 65-100 critique.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="small-card">
                <strong>Formule du score</strong>
                <span>EcoTraffic Score = 62% pollution + 38% trafic. Le trafic reel est utilise si traffic_cleaned.csv est disponible.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle:
        st.subheader("Signal pollution")
        trend = filtered_df.copy()
        trend["pollutant_smoothed"] = trend[pollutant].rolling(smooth_window, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend["datetime"],
                y=trend[pollutant],
                mode="lines",
                name="Mesure",
                line=dict(color="rgba(37, 99, 235, 0.35)", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trend["datetime"],
                y=trend["pollutant_smoothed"],
                mode="lines",
                name="Tendance",
                line=dict(color="#0CAF6D", width=3),
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=12, r=12, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827"),
            legend=dict(orientation="h"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Decomposition")
        st.plotly_chart(gauge("Pollution", pollution_score), use_container_width=True)
        st.plotly_chart(gauge("Trafic", traffic_score), use_container_width=True)
        st.markdown(
            """
            <div class="small-card">
                <strong>Limites a connaitre</strong>
                <span>Le modele predit uniquement le NO2. Les donnees trafic et environnement peuvent avoir des dates differentes selon les collectes.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        left, right = st.columns([1.25, 1])

        with left:
            fig_speed = px.line(
                traffic_view.groupby("datetime_hour", as_index=False)
                .agg(avg_speed=("Vitesse Moyenne (km/h)", "mean")),
                x="datetime_hour",
                y="avg_speed",
                markers=True,
                labels={"datetime_hour": "Heure", "avg_speed": "Vitesse moyenne (km/h)"},
            )
            fig_speed.update_layout(
                height=360,
                margin=dict(l=8, r=8, t=12, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                font=dict(color="#111827"),
            )
            st.plotly_chart(fig_speed, use_container_width=True)

        with right:
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
    c3.markdown('<div class="small-card"><strong>Valeur demo</strong><span>Une lecture geographique claire pour la soutenance et le pilotage.</span></div>', unsafe_allow_html=True)

with tab_model:
    st.subheader("Performance du modele NO2")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", metrics["MAE"])
    m2.metric("RMSE", metrics["RMSE"])
    m3.metric("R2", metrics["R2"])
    m4.metric("Algorithme", "XGBoost")

    left, right = st.columns([1, 1])
    with left:
        fig_importance = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#2563EB", "#0CAF6D", "#D97706"],
            labels={"importance": "Importance", "feature": "Variable"},
        )
        fig_importance.update_layout(
            height=520,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    with right:
        if MODEL_IMAGE_PATH.exists():
            st.image(str(MODEL_IMAGE_PATH), use_container_width=True)
        else:
            st.info("L'image des resultats n'existe pas encore. Lance `python ml/env_model_viz.py`.")

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

    summary = build_decision_summary(
        status=status,
        score=score,
        pollution_score=pollution_score,
        traffic_score=traffic_score,
        traffic_source=str(traffic_source),
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
        st.subheader("Aide a la decision")
        for action in decision_actions(status):
            st.markdown(
                f"""
                <div class="small-card">
                    <strong>Action</strong>
                    <span>{action}</span>
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

with tab_mvp:
    st.subheader("Conformite MVP")
    st.markdown(
        """
        <div class="small-card">
            <strong>Objectif du MVP</strong>
            <span>Montrer une chaine data complete pour la mobilite urbaine: collecte, transformation, modele predictif, visualisation et aide a la decision.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mvp1, mvp2, mvp3, mvp4 = st.columns(4)
    mvp1.metric("Pipeline data", "OK")
    mvp2.metric("Open data", "3 sources")
    mvp3.metric("Dashboard", "Interactif")
    mvp4.metric("Modele ML", "NO2")

    checklist_left, checklist_right = st.columns(2)
    with checklist_left:
        st.subheader("Attendus couverts")
        covered_items = [
            "Collecte de donnees ouvertes: trafic, meteo, pollution.",
            "Nettoyage et preparation des donnees.",
            "Integration trafic reel Rennes avec coordonnees GPS.",
            "Modele predictif operationnel pour le NO2.",
            "Dashboard interactif avec carte, filtres et exports.",
            "Rapport decisionnel automatique.",
        ]
        for item in covered_items:
            st.markdown(
                f"""
                <div class="small-card">
                    <strong>OK</strong>
                    <span>{item}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with checklist_right:
        st.subheader("Perspectives")
        next_items = [
            "Predire le trafic avec un historique plus long.",
            "Colorer les geometries completes des routes sur la carte.",
            "Synchroniser automatiquement trafic et environnement sur la meme fenetre temporelle.",
            "Ajouter une API ou une application citoyenne.",
            "Industrialiser le deploiement avec CI/CD complet.",
        ]
        for item in next_items:
            st.markdown(
                f"""
                <div class="small-card">
                    <strong>Evolution</strong>
                    <span>{item}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_data:
    st.subheader("Qualite et couverture des donnees")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Lignes", f"{len(filtered_df)}")
    d2.metric("Debut", filtered_df["datetime"].min().strftime("%d/%m/%Y"))
    d3.metric("Fin", filtered_df["datetime"].max().strftime("%d/%m/%Y"))
    d4.metric("Valeurs manquantes", int(filtered_df.isna().sum().sum()))

    left, right = st.columns([1, 1])
    with left:
        missing = (
            filtered_df.isna()
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"index": "colonne", 0: "missing_pct"})
        )
        fig_missing = px.bar(missing, x="colonne", y="missing_pct", labels={"missing_pct": "% manquant"})
        fig_missing.update_layout(
            height=360,
            margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    with right:
        corr_cols = ["temperature_2m", "precipitation", "wind_kmh", "pm10", "pm2_5", "carbon_monoxide", TARGET, "ozone"]
        corr = filtered_df[corr_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1)
        fig_corr.update_layout(
            height=360,
            margin=dict(l=8, r=8, t=12, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.dataframe(filtered_df.tail(30), use_container_width=True, hide_index=True)

with tab_arch:
    st.subheader("Architecture cible")
    st.markdown(
        """
        <div class="small-card">
            <strong>Flux de bout en bout</strong>
            <span>APIs Rennes/Open-Meteo -> MongoDB -> Kedro -> CSV/PostgreSQL -> Modele ML -> Dashboard Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_arch = go.Figure()
    nodes = [
        ("APIs", 0.05, 0.65),
        ("MongoDB", 0.23, 0.65),
        ("Kedro", 0.41, 0.65),
        ("PostgreSQL / CSV", 0.61, 0.65),
        ("XGBoost", 0.79, 0.78),
        ("Streamlit", 0.79, 0.50),
    ]
    for label, x, y in nodes:
        fig_arch.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=58, color="#0CAF6D", line=dict(width=2, color="#FFFFFF")),
                text=[label],
                textposition="bottom center",
                textfont=dict(color="#111827", size=14),
                hoverinfo="skip",
            )
        )
    arrows = [(0.10, 0.20), (0.28, 0.38), (0.47, 0.57), (0.67, 0.75), (0.67, 0.75)]
    y_pairs = [(0.65, 0.65), (0.65, 0.65), (0.65, 0.65), (0.66, 0.76), (0.63, 0.52)]
    for (x0, x1), (y0, y1) in zip(arrows, y_pairs):
        fig_arch.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.3,
            arrowwidth=2,
            arrowcolor="#2563EB",
        )
    fig_arch.update_layout(
        height=360,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0.35, 0.9]),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
    )
    st.plotly_chart(fig_arch, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    a1.markdown('<div class="small-card"><strong>Ce qui est branche</strong><span>Donnees environnement, donnees trafic CSV, modele NO2, visualisations et simulation.</span></div>', unsafe_allow_html=True)
    a2.markdown('<div class="small-card"><strong>Evolution possible</strong><span>Ajouter les geometries exactes des routes pour colorer les axes sur la carte.</span></div>', unsafe_allow_html=True)
    a3.markdown('<div class="small-card"><strong>Message soutenance</strong><span>Le socle data/ML est pret, la couche dashboard rend le projet lisible et decisionnel.</span></div>', unsafe_allow_html=True)
