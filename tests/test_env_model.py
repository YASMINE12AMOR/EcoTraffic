"""
Tests du modèle ML environnement (prédiction NO2).
"""
import os
import pickle
import tempfile

import pandas as pd
import pytest

from ml.env_model import (
    FEATURES,
    TARGET,
    cross_validate,
    evaluate,
    feature_importance,
    load_data,
    train,
)


@pytest.fixture(scope="module")
def trained():
    df = load_data()
    model, X_train, X_test, y_train, y_test = train(df)
    return model, X_train, X_test, y_train, y_test, df


# --- Chargement des données ---

def test_load_data_non_vide():
    df = load_data()
    assert not df.empty


def test_load_data_contient_features_et_target():
    df = load_data()
    for col in FEATURES + [TARGET]:
        assert col in df.columns, f"Colonne manquante : {col}"


def test_load_data_sans_nan_dans_features():
    df = load_data()
    assert df[FEATURES].isna().sum().sum() == 0


def test_load_data_sans_nan_dans_target():
    df = load_data()
    assert df[TARGET].isna().sum() == 0


# --- Entraînement ---

def test_train_renvoie_modele_et_splits(trained):
    model, X_train, X_test, y_train, y_test, _ = trained
    assert model is not None
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_train_split_80_20(trained):
    _, X_train, X_test, _, _, df = trained
    total = len(X_train) + len(X_test)
    assert total == len(df)
    assert abs(len(X_test) / total - 0.2) < 0.05


def test_train_features_correctes(trained):
    _, X_train, _, _, _, _ = trained
    assert list(X_train.columns) == FEATURES


# --- Métriques ---

def test_evaluate_renvoie_trois_metriques(trained):
    model, _, X_test, _, y_test, _ = trained
    metrics = evaluate(model, X_test, y_test)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics


def test_mae_inferieur_a_2(trained):
    model, _, X_test, _, y_test, _ = trained
    metrics = evaluate(model, X_test, y_test)
    assert metrics["MAE"] < 2.0, f"MAE trop élevée : {metrics['MAE']} µg/m³"


def test_r2_superieur_a_0_5(trained):
    model, _, X_test, _, y_test, _ = trained
    metrics = evaluate(model, X_test, y_test)
    assert metrics["R2"] > 0.5, f"R² trop faible : {metrics['R2']}"


def test_rmse_superieur_a_zero(trained):
    model, _, X_test, _, y_test, _ = trained
    metrics = evaluate(model, X_test, y_test)
    assert metrics["RMSE"] > 0


# --- Validation croisée ---

def test_cross_validate_renvoie_r2_mean_et_std(trained):
    model, _, _, _, _, df = trained
    cv = cross_validate(model, df)
    assert "R2_CV_mean" in cv
    assert "R2_CV_std" in cv


def test_cross_validate_r2_mean_entre_moins1_et_1(trained):
    model, _, _, _, _, df = trained
    cv = cross_validate(model, df)
    assert -1.0 <= cv["R2_CV_mean"] <= 1.0


# --- Importance des features ---

def test_feature_importance_contient_toutes_les_features(trained):
    model, _, _, _, _, _ = trained
    fi = feature_importance(model)
    assert list(fi["feature"]) == sorted(FEATURES, key=lambda f: -fi.set_index("feature").loc[f, "importance"])


def test_feature_importance_somme_egale_1(trained):
    model, _, _, _, _, _ = trained
    fi = feature_importance(model)
    assert abs(fi["importance"].sum() - 1.0) < 1e-5


def test_feature_importance_triee_decroissant(trained):
    model, _, _, _, _, _ = trained
    fi = feature_importance(model)
    assert fi["importance"].is_monotonic_decreasing


# --- Sauvegarde et chargement ---

def test_save_and_load_model(trained):
    model, _, X_test, _, y_test, _ = trained
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(model, f)
        with open(tmp_path, "rb") as f:
            loaded = pickle.load(f)
        y_pred_original = model.predict(X_test)
        y_pred_loaded   = loaded.predict(X_test)
        assert list(y_pred_original) == list(y_pred_loaded)
    finally:
        os.unlink(tmp_path)


# --- Prédiction ---

def test_predict_renvoie_float(trained):
    model, _, X_test, _, _, _ = trained
    sample = X_test.iloc[[0]]
    result = model.predict(sample)
    assert isinstance(float(result[0]), float)


def test_predict_dans_plage_realiste(trained):
    model, _, X_test, _, _, _ = trained
    y_pred = model.predict(X_test)
    assert all(0 <= p <= 100 for p in y_pred), "NO2 prédit hors plage réaliste"
