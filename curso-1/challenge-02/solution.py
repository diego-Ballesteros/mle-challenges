"""DSRP — Challenge 02: Regresión Usage_kWh.

Entrena un sklearn Pipeline para predecir el consumo energético
de una planta siderúrgica, lo evalúa localmente con un split
cronológico, lo reentrena en el dataset completo, lo serializa
y lo envía a la API del curso.

El Pipeline acepta el DataFrame crudo tal como el servidor lo envía
(columnas originales del CSV) y aplica todo el preprocesamiento
internamente usando únicamente primitivos de sklearn — sin clases
custom — para que `pickle.loads` funcione en el servidor sin
dependencias adicionales.
"""

from __future__ import annotations

import base64
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR: Path = Path(__file__).resolve().parent
TRAIN_CSV: Path = SCRIPT_DIR / "train.csv"
ENV_PATH: Path = SCRIPT_DIR / ".env"
VALIDATION_FRACTION: float = 0.20
API_URL: str = (
    "https://utpyawapnk.execute-api.us-east-1.amazonaws.com"
    "/Prod/challenges/02/submissions"
)

# NSM (seconds from midnight) reemplaza a `hour` (correlación 1.00 según EDA)
# y evita tener que parsear la columna `date` con un transformer custom — lo
# que rompería pickle.loads en el servidor (no tiene nuestra clase ni cloudpickle).
NUMERICAL_FEATURES: list[str] = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "Leading_Current_Reactive_Power_kVarh",
    "NSM",
]
CATEGORICAL_FEATURES: list[str] = ["Load_Type", "WeekStatus", "Day_of_week"]


def build_pipeline() -> Pipeline:
    """Construye el Pipeline: ColumnTransformer + GradientBoostingRegressor.

    Solo usa transformers nativos de sklearn — el servidor puede
    deserializar con pickle.loads sin necesidad de cloudpickle ni
    de importar clases definidas en nuestro __main__.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carga train.csv. Devuelve (X, y, parsed_dates) para el split cronológico."""
    df = pd.read_csv(csv_path)
    parsed_dates = pd.to_datetime(df["date"], dayfirst=True)
    print(f"Dataset shape       : {df.shape}")
    print(
        f"Rango de fechas     : "
        f"{parsed_dates.min().date()} -> {parsed_dates.max().date()}"
    )
    y = df["Usage_kWh"].astype(float)
    X = df.drop(columns=["Usage_kWh"])
    return X, y, parsed_dates


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    parsed_dates: pd.Series,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide cronológicamente: el último `validation_fraction` queda como validación."""
    order = parsed_dates.sort_values().index
    n_val = int(round(len(order) * validation_fraction))
    val_idx = order[-n_val:]
    train_idx = order[:-n_val]
    split_date = parsed_dates.loc[val_idx].min()

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]
    print(f"Train size          : {len(X_train)} filas (antes de {split_date})")
    print(f"Validation size     : {len(X_val)} filas (desde {split_date})")
    return X_train, X_val, y_train, y_val


def evaluate(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
    """Predice sobre X_val e imprime RMSE / MAE / R^2."""
    preds = pipeline.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    r2 = float(r2_score(y_val, preds))

    print()
    print("================================")
    print("  Evaluacion local (validacion)")
    print("================================")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R^2  : {r2:.4f}")
    print("================================")

    rmse_ok = "PASS" if rmse < 10 else "FAIL"
    r2_ok = "PASS" if r2 > 0.90 else "FAIL"
    print(f"  RMSE < 10    -> {rmse_ok}")
    print(f"  R^2  > 0.90  -> {r2_ok}")
    print("================================")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def serialize_pipeline(pipeline: Pipeline) -> str:
    """Pickle + base64. Devuelve string listo para enviar en JSON."""
    raw = pickle.dumps(pipeline)
    encoded = base64.b64encode(raw).decode("utf-8")
    print()
    print("Pipeline serializado correctamente")
    print(f"Tamano              : {len(raw) / 1024:.2f} KB")
    return encoded


def submit(model_b64: str, api_key: str) -> dict[str, Any]:
    """Envía el modelo al endpoint del curso. Devuelve el JSON de respuesta."""
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {"model_b64": model_b64}
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    except requests.exceptions.Timeout as exc:
        raise SystemExit("Timeout - servidor no respondio en 60s") from exc
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit(f"No se pudo conectar: {exc}") from exc

    if response.status_code == 401:
        raise SystemExit("API key invalida")
    if response.status_code == 422:
        raise SystemExit(f"Error de validacion: {response.text}")
    if not response.ok:
        raise SystemExit(f"HTTP {response.status_code}: {response.text}")

    return response.json()


def print_server_response(payload: dict[str, Any]) -> None:
    """Imprime la respuesta del servidor con el formato del enunciado."""
    submission_id = payload.get("submission_id", "-")
    is_best = payload.get("is_best", False)
    last = payload.get("last_metrics", {}) or {}
    best = payload.get("best_metrics", {}) or {}

    def _fmt(value: Any, decimals: int = 2) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}"
        return str(value) if value is not None else "-"

    print()
    print("================================================")
    print("  DSRP - Challenge 02: Submission Result")
    print("================================================")
    print(f"  submission_id  : {submission_id}")
    print(f"  is_best        : {is_best}")
    print(f"  RMSE (server)  : {_fmt(last.get('rmse'))}")
    print(f"  MAE  (server)  : {_fmt(last.get('mae'))}")
    print(f"  R^2  (server)  : {_fmt(last.get('r2'), 4)}")
    print("------------------------------------------------")
    print("  Best submission so far:")
    print(f"  RMSE (best)    : {_fmt(best.get('rmse'))}")
    print(f"  MAE  (best)    : {_fmt(best.get('mae'))}")
    print(f"  R^2  (best)    : {_fmt(best.get('r2'), 4)}")
    print("================================================")


def main() -> None:
    """Orquesta carga, evaluación, reentrenamiento, serialización y submission."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("DSRP_API_KEY")
    if not api_key:
        raise SystemExit("DSRP_API_KEY no esta configurada en .env")

    X_full, y_full, parsed_dates = load_data(TRAIN_CSV)
    X_train, X_val, y_train, y_val = chronological_split(
        X_full, y_full, parsed_dates, VALIDATION_FRACTION
    )

    print()
    print("Entrenando pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_val, y_val)

    if metrics["rmse"] >= 10 or metrics["r2"] <= 0.90:
        raise SystemExit("Metricas locales no cumplen el minimo; aborto submission.")

    print()
    print("Reentrenando en dataset completo antes de enviar...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X_full, y_full)

    model_b64 = serialize_pipeline(final_pipeline)

    print()
    print("Enviando submission...")
    response = submit(model_b64, api_key)
    print_server_response(response)


if __name__ == "__main__":
    main()
