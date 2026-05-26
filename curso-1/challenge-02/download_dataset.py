"""Download the Challenge 02 dataset from the DSRP API."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://utpyawapnk.execute-api.us-east-1.amazonaws.com/Prod/challenges/02/dataset"
OUTPUT_PATH = Path(__file__).parent / "train.csv"
TIMEOUT = 30


def get_dataset_url(api_key: str) -> str:
    """Fetch the presigned train_url from the DSRP API.

    Args:
        api_key: The API key for the DSRP service.

    Returns:
        The presigned URL pointing to the train CSV file.

    Raises:
        SystemExit: On any HTTP or network error.
    """
    print("Obteniendo URL del dataset...")
    try:
        response = requests.get(
            API_URL,
            headers={"x-api-key": api_key},
            timeout=TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit(f"Error de conexión al API: {exc}") from exc
    except requests.exceptions.Timeout:
        raise SystemExit(f"Timeout tras {TIMEOUT}s esperando respuesta del API.")

    if response.status_code == 401:
        raise SystemExit("Error 401: API key inválida o no autorizada.")
    if response.status_code == 422:
        raise SystemExit(f"Error 422: Entidad no procesable — {response.text}")
    if not response.ok:
        raise SystemExit(f"Error {response.status_code}: {response.text}")

    data = response.json()
    train_url: str = data["train_url"]
    return train_url


def download_csv(url: str, output_path: Path) -> tuple[int, int, list[str]]:
    """Download a CSV from a URL and save it locally.

    Args:
        url: The URL to download the CSV from.
        output_path: Local path where the CSV will be saved.

    Returns:
        Tuple of (rows, columns, column_names).

    Raises:
        SystemExit: On any HTTP or network error.
    """
    import pandas as pd

    print("Descargando train.csv...")
    try:
        response = requests.get(url, timeout=120)
    except requests.exceptions.ConnectionError as exc:
        raise SystemExit(f"Error de conexión al descargar CSV: {exc}") from exc
    except requests.exceptions.Timeout:
        raise SystemExit("Timeout al descargar el CSV.")

    if response.status_code == 401:
        raise SystemExit("Error 401: No autorizado para descargar el archivo.")
    if response.status_code == 422:
        raise SystemExit(f"Error 422: URL de descarga inválida — {response.text}")
    if not response.ok:
        raise SystemExit(f"Error {response.status_code} al descargar CSV: {response.text}")

    output_path.write_bytes(response.content)

    df = pd.read_csv(output_path)
    return len(df), len(df.columns), list(df.columns)


def main() -> None:
    """Orchestrate dataset download for Challenge 02."""
    api_key = os.getenv("DSRP_API_KEY", "")
    if not api_key:
        raise SystemExit("DSRP_API_KEY no está definida en el archivo .env")

    train_url = get_dataset_url(api_key)
    rows, cols, columns = download_csv(train_url, OUTPUT_PATH)

    print(f"Dataset guardado: {rows} filas x {cols} columnas")
    print(f"Columnas: {columns}")


if __name__ == "__main__":
    main()
