"""
Tests de la comparaison de 6 modeles ML (compare_models.py).
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from ml.compare_models import (
    COMPARISON_PATH,
    cross_validate_model,
    evaluate_model,
    train_and_evaluate,
)
from ml.env_model import FEATURES, TARGET, load_data


@pytest.fixture(scope="module")
def data_splits():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, X, y


@pytest.fixture(scope="module")
def trained_rf(data_splits):
    X_train, X_test, y_train, y_test, _, _ = data_splits
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


# --- evaluate_model ---

def test_evaluate_model_renvoie_trois_metriques(trained_rf, data_splits):
    _, X_test, _, y_test, _, _ = data_splits
    result = evaluate_model(trained_rf, X_test, y_test, "RF")
    assert "MAE" in result
    assert "RMSE" in result
    assert "R2" in result


def test_evaluate_model_metriques_sont_des_floats(trained_rf, data_splits):
    _, X_test, _, y_test, _, _ = data_splits
    result = evaluate_model(trained_rf, X_test, y_test, "RF")
    assert isinstance(result["MAE"], float)
    assert isinstance(result["RMSE"], float)
    assert isinstance(result["R2"], float)


def test_evaluate_model_mae_positif(trained_rf, data_splits):
    _, X_test, _, y_test, _, _ = data_splits
    result = evaluate_model(trained_rf, X_test, y_test, "RF")
    assert result["MAE"] >= 0


def test_evaluate_model_rmse_positif(trained_rf, data_splits):
    _, X_test, _, y_test, _, _ = data_splits
    result = evaluate_model(trained_rf, X_test, y_test, "RF")
    assert result["RMSE"] >= 0


def test_evaluate_model_r2_entre_moins1_et_1(trained_rf, data_splits):
    _, X_test, _, y_test, _, _ = data_splits
    result = evaluate_model(trained_rf, X_test, y_test, "RF")
    assert -1.0 <= result["R2"] <= 1.0


def test_evaluate_model_rmse_superieur_ou_egal_mae(trained_rf, data_splits):
    _, X_test, _, y_test, _, _ = data_splits
    result = evaluate_model(trained_rf, X_test, y_test, "RF")
    assert result["RMSE"] >= result["MAE"] - 0.001


# --- cross_validate_model ---

def test_cross_validate_renvoie_cv_mean_et_std(trained_rf, data_splits):
    _, _, _, _, X, y = data_splits
    result = cross_validate_model(trained_rf, X, y, cv_folds=3)
    assert "CV_Mean" in result
    assert "CV_Std" in result


def test_cross_validate_cv_mean_est_float(trained_rf, data_splits):
    _, _, _, _, X, y = data_splits
    result = cross_validate_model(trained_rf, X, y, cv_folds=3)
    assert isinstance(result["CV_Mean"], float)


def test_cross_validate_cv_std_positif(trained_rf, data_splits):
    _, _, _, _, X, y = data_splits
    result = cross_validate_model(trained_rf, X, y, cv_folds=3)
    assert result["CV_Std"] >= 0


# --- train_and_evaluate ---

def test_train_and_evaluate_random_forest(data_splits):
    X_train, X_test, y_train, y_test, X, y = data_splits
    result = train_and_evaluate(
        RandomForestRegressor, "RF Test",
        {"n_estimators": 10, "random_state": 42},
        X_train, X_test, y_train, y_test, X, y,
        requires_scaling=False,
    )
    assert result is not None
    assert "MAE" in result
    assert "R2" in result
    assert "training_time_sec" in result
    assert "CV_Mean" in result


def test_train_and_evaluate_avec_scaling(data_splits):
    X_train, X_test, y_train, y_test, X, y = data_splits
    result = train_and_evaluate(
        Ridge, "Ridge Test",
        {"alpha": 1.0},
        X_train, X_test, y_train, y_test, X, y,
        requires_scaling=True,
    )
    assert result is not None
    assert "R2" in result


def test_train_and_evaluate_mesure_temps(data_splits):
    X_train, X_test, y_train, y_test, X, y = data_splits
    result = train_and_evaluate(
        RandomForestRegressor, "RF Timing",
        {"n_estimators": 5, "random_state": 42},
        X_train, X_test, y_train, y_test, X, y,
    )
    assert result["training_time_sec"] >= 0


# --- Fichier JSON de comparaison ---

def test_comparison_json_existe():
    assert COMPARISON_PATH.exists(), f"{COMPARISON_PATH} n'existe pas"


def test_comparison_json_est_valide():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_comparison_json_contient_6_modeles():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 6


def test_comparison_json_modeles_attendus():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    expected = {"XGBoost", "LightGBM", "CatBoost", "Random Forest", "Ridge Regression", "Neural Network"}
    assert set(data.keys()) == expected


def test_comparison_json_chaque_modele_a_metriques():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for name, entry in data.items():
        if entry is not None:
            assert "metrics" in entry, f"{name} n'a pas de metrics"
            assert "MAE" in entry["metrics"], f"{name} n'a pas de MAE"
            assert "RMSE" in entry["metrics"], f"{name} n'a pas de RMSE"
            assert "R2" in entry["metrics"], f"{name} n'a pas de R2"


def test_comparison_json_chaque_modele_a_justification():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for name, entry in data.items():
        if entry is not None:
            assert "justification" in entry, f"{name} n'a pas de justification"
            assert len(entry["justification"]) > 10


def test_comparison_json_chaque_modele_a_rang():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    ranks = []
    for name, entry in data.items():
        if entry is not None:
            assert "rank" in entry, f"{name} n'a pas de rank"
            ranks.append(entry["rank"])
    assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_comparison_json_xgboost_est_premier():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["XGBoost"]["rank"] == 1


def test_comparison_json_r2_decroissant_par_rang():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = [(e["rank"], e["metrics"]["R2"]) for e in data.values() if e is not None]
    entries.sort(key=lambda x: x[0])
    r2_values = [r2 for _, r2 in entries]
    assert r2_values == sorted(r2_values, reverse=True)
