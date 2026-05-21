"""
Modèle ML — Prédiction du NO2 (dioxyde d'azote) à Rennes.

Source des données : kedro_preprocessing/data/02_intermediate/preprocessed_env_df.csv
Target            : nitrogen_dioxide (µg/m³)
Algorithme        : XGBoost Regression

Pourquoi XGBoost ?
------------------
1. Petit dataset (120 lignes) : XGBoost gère bien les petits volumes sans overfitting
   grâce à la régularisation L1/L2 intégrée. Les réseaux de neurones (LSTM) nécessitent
   des milliers d'exemples pour converger correctement.

2. Features hétérogènes : le dataset mélange des variables continues (température, vent),
   des entiers (heure, jour) et des binaires (rain_flag, is_weekend). XGBoost n'exige
   aucune normalisation ni encodage préalable, contrairement à la régression linéaire
   ou aux SVR.

3. Relations non-linéaires : NO2 ne varie pas linéairement avec la température ou
   l'heure (pics aux heures de pointe, effet de la pluie). XGBoost capture ces
   interactions automatiquement via l'ensemble d'arbres de décision.

4. Interprétabilité : XGBoost fournit nativement l'importance des features, ce qui
   permet de comprendre POURQUOI le modèle prédit une valeur donnée (CO à 55%,
   ozone à 34%) — essentiel pour un projet environnemental.

5. Rapidité : entraînement en < 1 seconde sur ce volume, ce qui permet de ré-entraîner
   le modèle à chaque nouvelle collecte de données sans coût computationnel.

Alternatives écartées :
- Régression linéaire : relations non-linéaires dans les données → sous-ajustement
- Random Forest      : moins efficace que XGBoost sur petits datasets (variance élevée)
- LSTM               : nécessite beaucoup plus de données temporelles séquentielles
- SVR                : sensible à la normalisation et plus long à calibrer
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "kedro_preprocessing", "data", "02_intermediate", "preprocessed_env_df.csv",
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "env_no2_model.pkl")

FEATURES = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_kmh",
    "rain_flag",
    "carbon_monoxide",
    "ozone",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
]
TARGET = "nitrogen_dioxide"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def train(df: pd.DataFrame) -> tuple:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        "R2":   round(r2_score(y_test, y_pred), 3),
    }


def cross_validate(model, df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    y = df[TARGET]
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    return {
        "R2_CV_mean": round(scores.mean(), 3),
        "R2_CV_std":  round(scores.std(), 3),
    }


def feature_importance(model) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "feature":    FEATURES,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_model(model) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(input_data: dict) -> float:
    model = load_model()
    df = pd.DataFrame([input_data])[FEATURES]
    return round(float(model.predict(df)[0]), 2)


if __name__ == "__main__":
    df = load_data()
    print(f"Données chargées : {len(df)} lignes")

    model, X_train, X_test, y_train, y_test = train(df)
    print(f"Modèle entraîné  : {len(X_train)} train / {len(X_test)} test")

    metrics = evaluate(model, X_test, y_test)
    print("\n--- Métriques sur le jeu de test ---")
    print(f"  MAE  : {metrics['MAE']} µg/m³  (erreur moyenne)")
    print(f"  RMSE : {metrics['RMSE']} µg/m³  (erreur quadratique)")
    print(f"  R²   : {metrics['R2']}          (1.0 = parfait)")

    cv = cross_validate(model, df)
    print("\n--- Validation croisée (5-fold) ---")
    print(f"  R² moyen : {cv['R2_CV_mean']} ± {cv['R2_CV_std']}")

    print("\n--- Importance des features ---")
    print(feature_importance(model).to_string(index=False))

    save_model(model)
    print(f"\nModèle sauvegardé : {MODEL_PATH}")

    print("\n--- Exemple de prédiction ---")
    exemple = {
        "temperature_2m":  15.0,
        "precipitation":    0.0,
        "wind_speed_10m":   5.0,
        "wind_kmh":        18.0,
        "rain_flag":        0,
        "carbon_monoxide": 140.0,
        "ozone":           65.0,
        "hour":             8,
        "day_of_week":      0,
        "is_weekend":       0,
        "month":            5,
    }
    result = predict(exemple)
    print(f"  Conditions : lundi 8h, 15°C, pas de pluie, vent 5 m/s")
    print(f"  NO2 prédit : {result} µg/m³")
