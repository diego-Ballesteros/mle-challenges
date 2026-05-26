"""DSRP - Challenge 02: Regresion Usage_kWh (v2 - GBR tuneado + NSM).

Mejoras respecto a v1:
1. GradientBoostingRegressor con `n_estimators=600` y `subsample=0.85` (en v1
   eran 300 y 1.0). Mas arboles + subsampling estocastico = mejor generalizacion.
2. `make_date_attr_extractor` queda disponible como utilidad picklable (usa solo
   `pd.to_datetime`/`operator.attrgetter`/`functools.partial`/`np.reshape`,
   modulos que el servidor tiene), pero NO se enchufa al pipeline porque los
   experimentos mostraron que `month` es out-of-distribution en el holdout del
   servidor (train cubre Jan-Oct 2018, server test probablemente Nov-Dec) y
   degrada RMSE de 1.058 -> 1.158, y `hour` es redundante con NSM (correlacion 1.00).

Historico de intentos:
- v1: GBR + NSM, n_est=300                          -> server RMSE 1.058 (baseline)
- HistGBR + NSM + month (via stdlib date extractor) -> server RMSE 1.158 (month OOD)
- HistGBR + NSM + hour (sin month)                  -> server RMSE 1.179 (early stop overfit)

NOTAS sobre constraints del servidor (solo sklearn + pandas + numpy):
- Cualquier funcion/clase definida en `__main__` rompe pickle.loads del servidor
  (AttributeError: Can't get attribute ... on <module '__main__' from
  '/var/runtime/bootstrap.py'>).
- lightgbm, cloudpickle: ModuleNotFoundError -- no estan instalados.
- Funciones cuyo `__module__` apunte a pandas/numpy/operator/functools si resuelven.

Limite de payload: ~6 MB (AWS Lambda sync invocation).
Target: superar a student-15 (RMSE 0.9998).
"""

from __future__ import annotations

import base64
import operator
import os
import pickle
import sys
from functools import partial
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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

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
PREVIOUS_BEST_OURS: float = 1.0580
LEADERBOARD_TARGET: float = 0.9998  # student-15
MAX_MODEL_SIZE_KB: float = 6 * 1024  # AWS Lambda sync payload limit ~6 MB

NUMERICAL_FEATURES: list[str] = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "Leading_Current_Reactive_Power_kVarh",
    "NSM",
]
CATEGORICAL_FEATURES: list[str] = ["Load_Type", "WeekStatus", "Day_of_week"]


def make_date_attr_extractor(attr_chain: str) -> Pipeline:
    """Sub-pipeline picklable que extrae un atributo de fecha de la columna `date`.

    Encadena `pd.to_datetime` -> `operator.attrgetter(attr_chain)` -> `np.asarray`
    -> `np.reshape(..., (-1, 1))`. Todos los callables apuntan a modulos que el
    servidor tiene (`pandas`, `operator`, `numpy`, `functools`), asi que el
    pickle se deserializa sin necesidad de nuestra clase/funcion en `__main__`.

    `attr_chain` se aplica via `attrgetter` (e.g. `"dt.month"`).
    """
    return Pipeline(
        steps=[
            ("parse", FunctionTransformer(pd.to_datetime, kw_args={"dayfirst": True})),
            ("get", FunctionTransformer(operator.attrgetter(attr_chain))),
            ("arr", FunctionTransformer(np.asarray)),
            ("reshape", FunctionTransformer(partial(np.reshape, shape=(-1, 1)))),
        ]
    )


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: numericas escaladas + categoricas one-hot. Sin features de fecha."""
    return ColumnTransformer(
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


def build_model(n_estimators: int) -> GradientBoostingRegressor:
    """GradientBoostingRegressor tuneado: mas arboles + subsampling estocastico."""
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.85,
        random_state=42,
    )


def build_pipeline(n_estimators: int) -> Pipeline:
    """Pipeline final: ColumnTransformer -> GradientBoostingRegressor."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", build_model(n_estimators)),
        ]
    )


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carga train.csv. Devuelve (X, y, parsed_dates) para el split cronologico."""
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
    """Divide cronologicamente: el ultimo `validation_fraction` queda como validacion."""
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
    """Predice sobre X_val e imprime RMSE / MAE / R^2 + delta vs best previo."""
    preds = pipeline.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae = float(mean_absolute_error(y_val, preds))
    r2 = float(r2_score(y_val, preds))

    print()
    print("================================")
    print("  Evaluacion local (validacion)")
    print("================================")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R^2  : {r2:.6f}")
    print("================================")

    rmse_ok = "PASS" if rmse < 10 else "FAIL"
    r2_ok = "PASS" if r2 > 0.90 else "FAIL"
    print(f"  RMSE < 10    -> {rmse_ok}")
    print(f"  R^2  > 0.90  -> {r2_ok}")
    print("--------------------------------")
    print(f"  Previous best RMSE : {PREVIOUS_BEST_OURS:.4f}  (ours v1)")
    print(f"  Current RMSE       : {rmse:.4f}  (local val)")
    print(f"  Improvement        : {PREVIOUS_BEST_OURS - rmse:+.4f}")
    print(f"  Leaderboard target : {LEADERBOARD_TARGET:.4f}  (student-15)")
    print("================================")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def pickle_roundtrip_test(
    pipeline: Pipeline, X_sample: pd.DataFrame
) -> tuple[str, np.ndarray]:
    """Pickle -> base64 -> pickle.loads -> predict. Devuelve (b64, preds_loaded).

    Verifica que el blob serializado se puede deserializar con `pickle.loads`
    estandar (lo que hace el servidor). Si esto falla local, no tiene sentido
    enviar. OJO: que pase local NO garantiza que pase en el servidor, porque
    `__main__` aqui es este script y en el servidor es bootstrap.py.
    """
    raw = pickle.dumps(pipeline)
    b64 = base64.b64encode(raw).decode("utf-8")
    pipeline_loaded = pickle.loads(base64.b64decode(b64))
    preds_loaded = pipeline_loaded.predict(X_sample.head(5))
    print()
    print("Pickle round-trip OK")
    print(f"  Tamano serializado : {len(raw) / 1024:.2f} KB")
    print(f"  Sample predictions : {preds_loaded.tolist()}")
    return b64, preds_loaded


def submit(model_b64: str, api_key: str) -> dict[str, Any]:
    """Envia el modelo al endpoint del curso. Devuelve el JSON de respuesta."""
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
    """Imprime la respuesta del servidor con contexto del leaderboard."""
    submission_id = payload.get("submission_id", "-")
    is_best = payload.get("is_best", False)
    last = payload.get("last_metrics", {}) or {}
    best = payload.get("best_metrics", {}) or {}

    def _fmt(value: Any, decimals: int = 4) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}"
        return str(value) if value is not None else "-"

    server_rmse = last.get("rmse")

    print()
    print("================================================")
    print("  DSRP - Challenge 02: Submission Result")
    print("================================================")
    print(f"  submission_id      : {submission_id}")
    print(f"  is_best            : {is_best}")
    print(f"  Leaderboard target : {LEADERBOARD_TARGET:.4f}  (student-15)")
    print(f"  Our RMSE (server)  : {_fmt(server_rmse)}")
    if isinstance(server_rmse, (int, float)):
        delta = LEADERBOARD_TARGET - server_rmse
        verdict = "BEATS leader" if delta > 0 else "still behind"
        print(f"  Delta vs leader    : {delta:+.4f}  ({verdict})")
    print(f"  MAE  (server)      : {_fmt(last.get('mae'))}")
    print(f"  R^2  (server)      : {_fmt(last.get('r2'), 6)}")
    print("------------------------------------------------")
    print("  Best submission so far (ours):")
    print(f"  RMSE (best)        : {_fmt(best.get('rmse'))}")
    print(f"  MAE  (best)        : {_fmt(best.get('mae'))}")
    print(f"  R^2  (best)        : {_fmt(best.get('r2'), 6)}")
    print("================================================")


def main() -> None:
    """Carga -> split -> entrena -> evalua -> reentrena full -> roundtrip -> submit."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("DSRP_API_KEY")
    if not api_key:
        raise SystemExit("DSRP_API_KEY no esta configurada en .env")

    X_full, y_full, parsed_dates = load_data(TRAIN_CSV)
    X_train, X_val, y_train, y_val = chronological_split(
        X_full, y_full, parsed_dates, VALIDATION_FRACTION
    )

    n_estimators = 600
    print()
    print(f"Entrenando pipeline (GBR + NSM, n_estimators={n_estimators})...")
    pipeline = build_pipeline(n_estimators=n_estimators)
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_val, y_val)
    if metrics["rmse"] >= 10 or metrics["r2"] <= 0.90:
        raise SystemExit("Metricas locales no cumplen el minimo; aborto submission.")

    print()
    print("Reentrenando en dataset completo antes de enviar...")
    final_pipeline = build_pipeline(n_estimators=n_estimators)
    final_pipeline.fit(X_full, y_full)

    print()
    print("Test pickle round-trip antes de enviar...")
    model_b64, _ = pickle_roundtrip_test(final_pipeline, X_val)

    payload_kb = len(model_b64) / 1024
    print(f"  Payload b64 size  : {payload_kb:.0f} KB (limite {MAX_MODEL_SIZE_KB:.0f} KB)")
    if payload_kb > MAX_MODEL_SIZE_KB:
        raise SystemExit(
            f"Payload {payload_kb:.0f} KB > limite {MAX_MODEL_SIZE_KB:.0f} KB; "
            f"reduce n_estimators o num_leaves."
        )

    print()
    print("Enviando submission...")
    response = submit(model_b64, api_key)
    print_server_response(response)


if __name__ == "__main__":
    main()
