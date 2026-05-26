# Challenge 02 — Regresión

Predice el consumo eléctrico (`Usage_kWh`) de una planta siderúrgica usando datos operacionales y temporales.

## Qué hace este challenge

- Descarga el dataset de entrenamiento desde la API de DSRP.
- Realiza un análisis exploratorio completo (EDA) en `notebook.ipynb`.
- Objetivo mínimo: **RMSE < 10** y **R² > 0.90** sobre el conjunto de validación temporal.

## Setup

```bash
# 1. Instalar dependencias
uv sync

# 2. Configurar credenciales
cp .env.example .env
# Editar .env y añadir tu DSRP_API_KEY
```

## Uso

```bash
# Paso 1: Descargar el dataset
uv run python download_dataset.py

# Paso 2: Abrir el notebook de EDA
uv run jupyter notebook
```

## Estrategia de validación

Split temporal: los datos anteriores al **20 de octubre de 2018** se usan para entrenamiento y los posteriores para validación local. Esta estrategia refleja el orden real de los datos y evita data leakage.

## Métricas objetivo

| Métrica | Mínimo |
|---------|--------|
| RMSE    | < 10   |
| R²      | > 0.90 |
