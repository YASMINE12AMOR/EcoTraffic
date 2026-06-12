"""
Monitor model drift for the NO2 XGBoost model.
Compares current data distribution against training reference stats.
Saves a JSON report to ml/models/drift_report.json.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.env_model import load_data, load_model, FEATURES, TARGET

REPORT_PATH = Path(__file__).parent / "models" / "drift_report.json"
REFERENCE_STATS_PATH = Path(__file__).parent / "models" / "reference_stats.json"

DRIFT_THRESHOLDS = {
    "mean_shift_pct": 20.0,
    "std_shift_pct": 30.0,
    "mae_degradation_pct": 50.0,
}


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in FEATURES + [TARGET]:
        if col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "null_count": int(df[col].isna().sum()),
            }
    return stats


def save_reference_stats(df: pd.DataFrame) -> None:
    stats = compute_stats(df)
    REFERENCE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump({"created_at": datetime.now().isoformat(), "stats": stats}, f, indent=2)
    print(f"Reference stats saved to {REFERENCE_STATS_PATH}")


def detect_drift(reference: dict, current: dict) -> list[dict]:
    alerts = []
    for col, ref_vals in reference.items():
        if col not in current:
            continue
        cur_vals = current[col]
        if ref_vals["mean"] != 0:
            mean_shift = abs(cur_vals["mean"] - ref_vals["mean"]) / abs(ref_vals["mean"]) * 100
            if mean_shift > DRIFT_THRESHOLDS["mean_shift_pct"]:
                alerts.append({
                    "column": col,
                    "type": "mean_shift",
                    "reference_mean": ref_vals["mean"],
                    "current_mean": cur_vals["mean"],
                    "shift_pct": round(mean_shift, 2),
                })
        if ref_vals["std"] != 0:
            std_shift = abs(cur_vals["std"] - ref_vals["std"]) / abs(ref_vals["std"]) * 100
            if std_shift > DRIFT_THRESHOLDS["std_shift_pct"]:
                alerts.append({
                    "column": col,
                    "type": "std_shift",
                    "reference_std": ref_vals["std"],
                    "current_std": cur_vals["std"],
                    "shift_pct": round(std_shift, 2),
                })
    return alerts


def run_drift_report() -> dict:
    df = load_data()
    current_stats = compute_stats(df)

    report = {
        "generated_at": datetime.now().isoformat(),
        "n_rows": len(df),
        "drift_alerts": [],
        "prediction_check": {},
        "status": "OK",
    }

    if REFERENCE_STATS_PATH.exists():
        with open(REFERENCE_STATS_PATH) as f:
            ref_data = json.load(f)
        alerts = detect_drift(ref_data["stats"], current_stats)
        report["drift_alerts"] = alerts
        if alerts:
            report["status"] = "DRIFT_DETECTED"
            print(f"[WARN] {len(alerts)} drift alert(s) detected")
            for a in alerts:
                print(f"  - {a['column']}: {a['type']} ({a['shift_pct']}%)")
    else:
        print("No reference stats found. Saving current data as reference.")
        save_reference_stats(df)
        report["status"] = "REFERENCE_CREATED"

    model = load_model()
    X = df[FEATURES].dropna()
    y_true = df.loc[X.index, TARGET]
    y_pred = model.predict(X)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    report["prediction_check"] = {
        "mae": round(mae, 4),
        "n_samples": len(X),
        "no2_mean_current": round(float(y_true.mean()), 4),
        "no2_mean_predicted": round(float(y_pred.mean()), 4),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDrift report saved to {REPORT_PATH}")
    print(f"Status: {report['status']}")
    print(f"MAE current: {mae:.4f} µg/m³")
    return report


if __name__ == "__main__":
    run_drift_report()
