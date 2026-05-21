"""
Tests de qualité sur les données réelles météo + pollution.
Fichier source : kedro_preprocessing/data/02_intermediate/preprocessed_env_df.csv
"""
import os
import pandas as pd
import pytest

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "kedro_preprocessing", "data", "02_intermediate", "preprocessed_env_df.csv"
)

EXPECTED_COLUMNS = [
    "datetime", "temperature_2m", "precipitation", "wind_speed_10m",
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone",
    "sulphur_dioxide", "hour", "day_of_week", "is_weekend",
    "wind_kmh", "high_pm2_5", "high_pm10", "month", "rain_flag",
]


@pytest.fixture(scope="module")
def env_df():
    return pd.read_csv(DATA_PATH)


# --- Structure ---

def test_fichier_charge_correctement(env_df):
    assert not env_df.empty


def test_nombre_de_lignes(env_df):
    assert len(env_df) == 120


def test_colonnes_attendues_presentes(env_df):
    for col in EXPECTED_COLUMNS:
        assert col in env_df.columns, f"Colonne manquante : {col}"


# --- Qualité des données ---

def test_pas_de_valeurs_nulles_datetime(env_df):
    assert env_df["datetime"].isna().sum() == 0


def test_pas_de_doublons_datetime(env_df):
    assert env_df["datetime"].duplicated().sum() == 0


def test_donnees_triees_par_datetime_croissant(env_df):
    dates = pd.to_datetime(env_df["datetime"])
    assert list(dates) == sorted(dates.tolist())


# --- Plages de valeurs météo ---

def test_temperature_dans_plage_raisonnable(env_df):
    assert env_df["temperature_2m"].min() > -10
    assert env_df["temperature_2m"].max() < 50


def test_precipitation_positive_ou_nulle(env_df):
    assert env_df["precipitation"].min() >= 0


def test_vent_positif_ou_nul(env_df):
    assert env_df["wind_speed_10m"].min() >= 0


# --- Plages de valeurs pollution ---

def test_pm25_positive(env_df):
    assert env_df["pm2_5"].min() >= 0


def test_pm10_positive(env_df):
    assert env_df["pm10"].min() >= 0


def test_no2_positive(env_df):
    assert env_df["nitrogen_dioxide"].min() >= 0


def test_ozone_positif(env_df):
    assert env_df["ozone"].min() >= 0


# --- Cohérence des features calculées ---

def test_wind_kmh_egal_wind_ms_fois_3_6(env_df):
    expected = (env_df["wind_speed_10m"] * 3.6).round(4)
    actual = env_df["wind_kmh"].round(4)
    assert (expected == actual).all()


def test_high_pm25_coherent_avec_seuil_25(env_df):
    expected = env_df["pm2_5"] > 25
    actual = env_df["high_pm2_5"].astype(bool)
    assert (expected == actual).all()


def test_high_pm10_coherent_avec_seuil_50(env_df):
    expected = env_df["pm10"] > 50
    actual = env_df["high_pm10"].astype(bool)
    assert (expected == actual).all()


def test_rain_flag_coherent_avec_precipitation(env_df):
    expected = (env_df["precipitation"] > 0).astype(int)
    assert (env_df["rain_flag"] == expected).all()


def test_heures_entre_0_et_23(env_df):
    assert env_df["hour"].min() >= 0
    assert env_df["hour"].max() <= 23


def test_jour_semaine_entre_0_et_6(env_df):
    assert env_df["day_of_week"].min() >= 0
    assert env_df["day_of_week"].max() <= 6


def test_is_weekend_binaire(env_df):
    assert set(env_df["is_weekend"].unique()).issubset({0, 1})


def test_rain_flag_binaire(env_df):
    assert set(env_df["rain_flag"].unique()).issubset({0, 1})


def test_month_correspond_au_mois_datetime(env_df):
    dates = pd.to_datetime(env_df["datetime"])
    assert (dates.dt.month == env_df["month"]).all()
