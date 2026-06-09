"""Challenge 03 — Knapsack Logistics Optimizer submission script."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Carga las variables de entorno desde .env
BASE_URL = "https://utpyawapnk.execute-api.us-east-1.amazonaws.com/Prod"
CHALLENGE_PATH = "/challenges/03/submissions"
MAX_PROMPT_CHARS = 8000
POLL_INTERVAL = 5
MAX_POLL_ATTEMPTS = 40


def load_api_key() -> str:
    """Load and return the DSRP_API_KEY from the .env file.

    Returns:
        The API key string.

    Raises:
        SystemExit: If the key is missing or empty.
    """
    key = os.getenv("DSRP_API_KEY", "").strip()
    if not key:
        print("ERROR: DSRP_API_KEY no encontrada. Copia .env.example a .env y agrega tu clave.")
        sys.exit(1)
    return key


def load_prompt() -> str:
    """Read and return the prompt from prompt.txt.

    Returns:
        The prompt string.

    Raises:
        SystemExit: If the file is not found.
    """
    prompt_path = Path(__file__).parent / "prompt.txt"
    if not prompt_path.exists():
        print(f"ERROR: prompt.txt no encontrado en {prompt_path}")
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8")


def validate_prompt(prompt: str) -> None:
    """Validate prompt length and print char count.

    Args:
        prompt: The prompt string to validate.

    Raises:
        SystemExit: If the prompt exceeds MAX_PROMPT_CHARS.
    """
    count = len(prompt)
    print(f"Longitud del prompt: {count} caracteres")
    if count > MAX_PROMPT_CHARS:
        print(f"ERROR: El prompt excede el límite de {MAX_PROMPT_CHARS} caracteres ({count}).")
        sys.exit(1)


def submit_prompt(prompt: str, api_key: str) -> dict:
    """POST the prompt to the submissions endpoint.

    Args:
        prompt: The prompt string to submit.
        api_key: The DSRP API key for authentication.

    Returns:
        The parsed JSON response dict containing submission_id and poll path.

    Raises:
        SystemExit: On 401, 422, timeout, or connection error.
    """
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    payload = {"prompt": prompt}
    try:
        resp = requests.post(
            f"{BASE_URL}{CHALLENGE_PATH}",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except requests.Timeout:
        print("ERROR: Timeout en submission (30s).")
        sys.exit(1)
    except requests.ConnectionError as exc:
        print(f"ERROR: No se pudo conectar al servidor — {exc}")
        sys.exit(1)

    if resp.status_code == 401:
        print("ERROR: API key inválida (401).")
        sys.exit(1)
    if resp.status_code == 422:
        detail = resp.json().get("detail", resp.text)
        print(f"ERROR: Error de validación (422) — {detail}")
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def poll_result(submission_id: str, api_key: str) -> dict:
    """Poll until the submission is no longer in 'scoring' status.

    Args:
        submission_id: The ID returned from the submission endpoint.
        api_key: The DSRP API key for authentication.

    Returns:
        The final result dict once scoring is complete.
    """
    headers = {"x-api-key": api_key}
    url = f"{BASE_URL}{CHALLENGE_PATH}/{submission_id}"
    result: dict = {}

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        print(f"Esperando resultado... (intento {attempt}/{MAX_POLL_ATTEMPTS})")
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except requests.Timeout:
            print(f"  Timeout en polling (intento {attempt}), reintentando...")
            time.sleep(POLL_INTERVAL)
            continue
        except requests.RequestException as exc:
            print(f"  Error en polling: {exc}")
            break

        if result.get("status") != "scoring":
            break

        time.sleep(POLL_INTERVAL)

    return result


def print_result(result: dict, submission_id: str) -> None:
    """Print the formatted final result with leaderboard comparison.

    Args:
        result: The scored result dict from the API.
        submission_id: The submission ID string.
    """
    status = result.get("status", "unknown")
    metrics = result.get("metrics", {})
    value_ratio = metrics.get("value_ratio", float("nan"))
    exact_match = metrics.get("exact_match", float("nan"))
    n_instances = metrics.get("n", "?")

    leader_score = 0.9987
    try:
        delta = float(value_ratio) - leader_score
        delta_str = f"{delta:+.4f}"
    except (TypeError, ValueError):
        delta_str = "N/A"

    print("================================================")
    print("  DSRP — Challenge 03: Knapsack Result")
    print("================================================")
    print(f"  submission_id  : {submission_id}")
    print(f"  status         : {status}")
    print(f"  value_ratio    : {value_ratio}")
    print(f"  exact_match    : {exact_match}")
    print(f"  n_instances    : {n_instances}")
    print("------------------------------------------------")
    print(f"  Leaderboard target : {leader_score} (student-01)")
    print(f"  value_ratio obtenido: {value_ratio}")
    print(f"  Delta vs leader    : {delta_str}")
    print("================================================")


def main() -> None:
    """Run the full submit-and-poll flow for Challenge 03."""
    api_key = load_api_key()
    prompt = load_prompt()
    validate_prompt(prompt)

    # Verificación: el prompt se envía TAL CUAL desde prompt.txt; el servidor
    # sustituye {{input}}, no este script. Mostramos lo que realmente se manda.
    print("------------------------------------------------")
    print("Prompt exacto enviado (primeros 500 chars):")
    print(repr(prompt[:500]))
    print(f"Contiene placeholder '{{{{input}}}}': {'{{input}}' in prompt}")
    print("------------------------------------------------")

    print("Enviando prompt a la API...")
    submission = submit_prompt(prompt, api_key)

    submission_id = submission.get("submission_id", "")
    status = submission.get("status", "")
    print(f"Submission recibida — ID: {submission_id}, estado: {status}")

    if status == "scoring":
        result = poll_result(submission_id, api_key)
    else:
        result = submission

    print_result(result, submission_id)


if __name__ == "__main__":
    main()
