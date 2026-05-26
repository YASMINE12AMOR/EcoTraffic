"""
Tests de qualité sur les données réelles de trafic.
Fichier source : ecotraffic/data/03_primary/traffic_cleaned.csv
"""
import os
import pandas as pd
import pytest

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "ecotraffic", "data", "03_primary", "traffic_cleaned.csv"
)

VALID_STATUTS = {"freeflow", "heavy", "congested", "unknown"}
VALID_HIERARCHIES = {
    "Réseau de distribution principale",
    "Réseau d'appui",
    "Réseau d'armature",
    "Réseau national",
}
EXPECTED_COLUMNS = [
    "Date/Heure",
    "Route",
    "Vitesse Moyenne (km/h)",
    "Vitesse Max (km/h)",
    "Temps de Trajet (s)",
    "Fiabilité (%)",
    "Statut Trafic",
    "Hiérarchie",
    "_id",
    "datetime_hour",
]


@pytest.fixture(scope="module")
def traffic_df():
    return pd.read_csv(DATA_PATH)


# --- Structure ---

def test_fichier_charge_correctement(traffic_df):
    assert not traffic_df.empty


def test_nombre_de_lignes(traffic_df):
    assert len(traffic_df) > 0


def test_colonnes_attendues_presentes(traffic_df):
    for col in EXPECTED_COLUMNS:
        assert col in traffic_df.columns, f"Colonne manquante : {col}"


# --- Qualité des données ---

def test_pas_de_valeurs_nulles_date(traffic_df):
    assert traffic_df["Date/Heure"].isna().sum() == 0


def test_pas_de_valeurs_nulles_route(traffic_df):
    assert traffic_df["Route"].isna().sum() == 0


def test_pas_de_valeurs_nulles_vitesse(traffic_df):
    assert traffic_df["Vitesse Moyenne (km/h)"].isna().sum() == 0


def test_pas_de_doublons(traffic_df):
    assert traffic_df.duplicated().sum() == 0


# --- Plages de valeurs ---

def test_vitesse_moyenne_entre_0_et_130(traffic_df):
    assert traffic_df["Vitesse Moyenne (km/h)"].min() >= 0
    assert traffic_df["Vitesse Moyenne (km/h)"].max() <= 130


def test_vitesse_max_positive(traffic_df):
    assert traffic_df["Vitesse Max (km/h)"].min() > 0


def test_temps_de_trajet_positif_ou_nul(traffic_df):
    assert traffic_df["Temps de Trajet (s)"].min() >= 0


def test_fiabilite_entre_0_et_100(traffic_df):
    assert traffic_df["Fiabilité (%)"].min() >= 0
    assert traffic_df["Fiabilité (%)"].max() <= 100


# --- Valeurs catégorielles ---

def test_statuts_trafic_valides(traffic_df):
    statuts_observes = set(traffic_df["Statut Trafic"].dropna().unique())
    assert statuts_observes.issubset(VALID_STATUTS), (
        f"Statuts inattendus : {statuts_observes - VALID_STATUTS}"
    )


def test_statuts_en_minuscules(traffic_df):
    assert traffic_df["Statut Trafic"].str.islower().all()


def test_hierarchies_valides(traffic_df):
    hierarchies_observees = set(traffic_df["Hiérarchie"].dropna().unique())
    assert hierarchies_observees.issubset(VALID_HIERARCHIES), (
        f"Hiérarchies inattendues : {hierarchies_observees - VALID_HIERARCHIES}"
    )


def test_routes_non_vides(traffic_df):
    assert (traffic_df["Route"].str.strip() != "").all()


# --- Cohérence temporelle ---

def test_colonne_datetime_hour_existe(traffic_df):
    assert "datetime_hour" in traffic_df.columns


def test_datetime_hour_coherent_avec_date_heure(traffic_df):
    dates = pd.to_datetime(traffic_df["Date/Heure"], utc=True)
    hours = pd.to_datetime(traffic_df["datetime_hour"], utc=True)
    assert (dates.dt.floor("h") == hours).all()
