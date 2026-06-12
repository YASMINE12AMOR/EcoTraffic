"""
Visualisation des résultats du modèle NO2.
Lancer : python ml/env_model_viz.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from ml.env_model import (  # noqa: E402
    FEATURES,
    evaluate,
    feature_importance,
    load_data,
    train,
)

# ── Entraînement ────────────────────────────────────────────────────────────
df = load_data()
model, X_train, X_test, y_train, y_test = train(df)
y_pred = model.predict(X_test)
metrics = evaluate(model, X_test, y_test)
fi = feature_importance(model)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Modèle XGBoost — Prédiction NO₂ à Rennes", fontsize=15, fontweight="bold")

# ── Graphique 1 : Réel vs Prédit ─────────────────────────────────────────
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred, color="#2196F3", alpha=0.8, edgecolors="white", s=80)
lim = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
ax1.plot(lim, lim, "r--", linewidth=1.5, label="Prédiction parfaite")
ax1.set_xlabel("NO₂ réel (µg/m³)")
ax1.set_ylabel("NO₂ prédit (µg/m³)")
ax1.set_title("Réel vs Prédit")
ax1.legend()
ax1.text(0.05, 0.90, f"R² = {metrics['R2']}\nMAE = {metrics['MAE']} µg/m³",
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# ── Graphique 2 : Erreurs de prédiction ─────────────────────────────────
ax2 = axes[0, 1]
errors = y_pred - y_test.values
ax2.bar(range(len(errors)), errors,
        color=["#F44336" if e > 0 else "#4CAF50" for e in errors], alpha=0.8)
ax2.axhline(0, color="black", linewidth=1)
ax2.set_xlabel("Échantillon de test")
ax2.set_ylabel("Erreur (prédit − réel) µg/m³")
ax2.set_title("Erreurs de prédiction")
ax2.text(0.05, 0.92, f"RMSE = {metrics['RMSE']} µg/m³",
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# ── Graphique 3 : Importance des features ───────────────────────────────
ax3 = axes[1, 0]
colors = ["#1976D2" if i < 3 else "#90CAF9" for i in range(len(fi))]
bars = ax3.barh(fi["feature"], fi["importance"] * 100, color=colors, edgecolor="white")
ax3.set_xlabel("Importance (%)")
ax3.set_title("Importance des features")
ax3.invert_yaxis()
for bar, val in zip(bars, fi["importance"] * 100):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=9)

# ── Graphique 4 : NO₂ prédit par heure de la journée ────────────────────
ax4 = axes[1, 1]
heures = list(range(24))
base = {
    "temperature_2m": 15.0, "precipitation": 0.0,
    "wind_speed_10m": 5.0, "wind_kmh": 18.0,
    "rain_flag": 0, "carbon_monoxide": 140.0,
    "ozone": 65.0, "day_of_week": 0,
    "is_weekend": 0, "month": 5,
}
scenarios = {
    "Lundi (jour de semaine)": {**base, "day_of_week": 0, "is_weekend": 0},
    "Samedi (weekend)": {**base, "day_of_week": 5, "is_weekend": 1},
    "Jour de pluie": {**base, "day_of_week": 0, "precipitation": 1.5, "rain_flag": 1},
}
colors_sc = ["#1976D2", "#FF9800", "#4CAF50"]
for (label, params), color in zip(scenarios.items(), colors_sc):
    preds = []
    for h in heures:
        row = pd.DataFrame([{**params, "hour": h}])[FEATURES]
        preds.append(float(model.predict(row)[0]))
    ax4.plot(heures, preds, marker="o", markersize=4, label=label, color=color, linewidth=2)

ax4.set_xlabel("Heure de la journée")
ax4.set_ylabel("NO₂ prédit (µg/m³)")
ax4.set_title("NO₂ prédit selon l'heure — 3 scénarios")
ax4.set_xticks(range(0, 24, 2))
ax4.axvspan(7, 9, alpha=0.1, color="red", label="Heure de pointe matin")
ax4.axvspan(17, 19, alpha=0.1, color="orange", label="Heure de pointe soir")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ml/models/env_model_results.png", dpi=150, bbox_inches="tight")
print("Graphique sauvegardé : ml/models/env_model_results.png")
plt.show()
