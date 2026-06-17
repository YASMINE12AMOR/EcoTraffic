"""
Comparaison de 6 modèles ML pour la prédiction de NO2.

Ce script teste différents algorithmes et stocke les résultats dans
ml/models/models_comparison.json pour utilisation dans Streamlit.
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.env_model import FEATURES, TARGET, load_data

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from sklearn.neural_network import MLPRegressor
except ImportError:
    MLPRegressor = None

COMPARISON_PATH = Path(__file__).parent / "models" / "models_comparison.json"


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Évalue les performances d'un modèle."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 3),
    }


def cross_validate_model(model, X, y, cv_folds: int = 5) -> dict:
    """Cross-validation K-Fold."""
    try:
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2", n_jobs=-1)
        return {
            "CV_Mean": round(float(scores.mean()), 3),
            "CV_Std": round(float(scores.std()), 3),
        }
    except Exception as e:
        print(f"  Erreur CV: {e}")
        return {"CV_Mean": None, "CV_Std": None}


def train_and_evaluate(
    model_class,
    model_name: str,
    model_params: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    X,
    y,
    requires_scaling: bool = False,
) -> dict:
    """Entraîne et évalue un modèle."""
    print(f"\n  {model_name}...", end=" ", flush=True)
    start_time = time.time()

    try:
        # Copie les données en cas de scaling
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_scaled = X.copy()
        
        if requires_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_scaled = scaler.fit_transform(X)

        # Entraînement
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)

        # Évaluation
        metrics = evaluate_model(model, X_test_scaled, y_test, model_name)
        cv_metrics = cross_validate_model(model, X_scaled, y, cv_folds=5)
        
        training_time = time.time() - start_time
        metrics["training_time_sec"] = round(training_time, 3)
        metrics.update(cv_metrics)

        print(f"OK ({training_time:.2f}s) | MAE={metrics['MAE']} | R2={metrics['R2']}")
        return metrics

    except Exception as e:
        print(f"ERREUR: {str(e)[:50]}")
        return None


def compare_models():
    """Compare 6 modèles ML."""
    print("\n" + "=" * 80)
    print("COMPARAISON DE 6 MODÈLES ML POUR LA PRÉDICTION DE NO2")
    print("=" * 80)

    # Chargement des données
    print("\n[*] Chargement des donnees...")
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   {len(df)} samples | {len(X_train)} train | {len(X_test)} test")

    # Modèles à tester avec justifications
    models_to_test = {
        "XGBoost": {
            "available": XGBRegressor is not None,
            "class": XGBRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": 42,
                "verbosity": 0,
            },
            "scaling": False,
            "justification": "Gradient boosting classique. Gère bien les petits datasets et relations non-linéaires.",
        },
        "LightGBM": {
            "available": LGBMRegressor is not None,
            "class": LGBMRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "num_leaves": 20,
                "random_state": 42,
                "verbose": -1,
            },
            "scaling": False,
            "justification": "Plus rapide que XGBoost. Meilleur sur petits datasets, moins d'overfitting.",
        },
        "CatBoost": {
            "available": CatBoostRegressor is not None,
            "class": CatBoostRegressor,
            "params": {
                "iterations": 100,
                "depth": 4,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": False,
                "allow_writing_files": False,
            },
            "scaling": False,
            "justification": "Conçu pour données hétérogènes. Pas besoin d'encodage, gère bien les features mixtes.",
        },
        "Random Forest": {
            "available": True,
            "class": RandomForestRegressor,
            "params": {
                "n_estimators": 100,
                "max_depth": 4,
                "min_samples_split": 3,
                "random_state": 42,
                "n_jobs": -1,
            },
            "scaling": False,
            "justification": "Ensemble simple. Baseline pour comparaison. Moins d'overfitting que single tree.",
        },
        "Ridge Regression": {
            "available": True,
            "class": Ridge,
            "params": {"alpha": 1.0, "random_state": 42},
            "scaling": True,
            "justification": "Modèle linéaire simple. Requiert scaling. Performant si relations sont linéaires.",
        },
        "Neural Network": {
            "available": MLPRegressor is not None,
            "class": MLPRegressor,
            "params": {
                "hidden_layer_sizes": (16, 8),
                "max_iter": 500,
                "learning_rate_init": 0.01,
                "random_state": 42,
                "early_stopping": True,
                "validation_fraction": 0.1,
            },
            "scaling": True,
            "justification": "Réseau de neurones petit (pour petit dataset). Demande normalization.",
        },
    }

    results = {}

    # Test de chaque modèle
    print("\n[*] Entrainement des modeles...")
    for model_name, model_config in models_to_test.items():
        if not model_config["available"]:
            print(f"  {model_name}... [!] Dependance manquante")
            results[model_name] = None
            continue

        metrics = train_and_evaluate(
            model_config["class"],
            model_name,
            model_config["params"],
            X_train,
            X_test,
            y_train,
            y_test,
            X,
            y,
            requires_scaling=model_config["scaling"],
        )
        
        if metrics:
            results[model_name] = {
                "metrics": metrics,
                "justification": model_config["justification"],
            }
        else:
            results[model_name] = None

    # Classement par R²
    print("\n" + "=" * 80)
    print("CLASSEMENT PAR R2 (Score de determination)")
    print("=" * 80)
    
    ranked = sorted(
        [(name, r["metrics"]["R2"]) for name, r in results.items() if r is not None],
        key=lambda x: x[1],
        reverse=True,
    )
    
    for rank, (name, r2_score) in enumerate(ranked, 1):
        medal = {1: "[1er]", 2: "[2e]", 3: "[3e]"}.get(rank, "    ")
        print(f"{medal} #{rank}. {name:20} R2 = {r2_score:.3f}")
    
    # Meilleur modele
    if ranked:
        best_model, best_r2 = ranked[0]
        print("\n" + "=" * 80)
        print(f">>> MEILLEUR MODELE: {best_model} (R2 = {best_r2:.3f})")
        print("=" * 80)

    # Sauvegarde résultats
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Conversion pour JSON serializable
    json_results = {}
    for name, result in results.items():
        if result:
            json_results[name] = {
                "metrics": result["metrics"],
                "justification": result["justification"],
                "rank": next((i+1 for i, (n, _) in enumerate(ranked) if n == name), None)
            }
        else:
            json_results[name] = None
    
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Resultats sauvegardes : {COMPARISON_PATH}")
    print("\n")


if __name__ == "__main__":
    compare_models()
