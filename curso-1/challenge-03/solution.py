"""Challenge 03 — Knapsack Logistics Optimizer submission script."""

from __future__ import annotations

import argparse
import math
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
MAX_POLL_ATTEMPTS = 60
REPEAT_DELAY = 30  # segundos de espera entre submissions en modo --repeat


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
    metrics = result.get("metrics") or {}
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


def run_single(prompt: str, api_key: str) -> tuple[str, dict]:
    """Submit the prompt once and poll until it is scored.

    Args:
        prompt: The prompt string to submit.
        api_key: The DSRP API key for authentication.

    Returns:
        A tuple of (submission_id, final result dict).
    """
    print("Enviando prompt a la API...")
    submission = submit_prompt(prompt, api_key)

    submission_id = submission.get("submission_id", "")
    status = submission.get("status", "")
    print(f"Submission recibida — ID: {submission_id}, estado: {status}")

    if status == "scoring":
        result = poll_result(submission_id, api_key)
    else:
        result = submission

    return submission_id, result


def _fmt_metric(value: object) -> str:
    """Format a metric as a 4-decimal string, or pass through if non-numeric."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def print_summary_table(rows: list[dict]) -> None:
    """Print a box-drawing summary table plus best/mean value_ratio.

    Args:
        rows: One dict per run with keys n, submission_id, value_ratio, exact_match.
    """
    headers = ["#", "submission_id", "value_ratio", "exact_match"]
    table = [
        [
            str(r["n"]),
            r["submission_id"] or "—",
            _fmt_metric(r["value_ratio"]),
            _fmt_metric(r["exact_match"]),
        ]
        for r in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table))
        for i in range(len(headers))
    ]

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def row(cells: list[str]) -> str:
        return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, widths)) + " │"

    print(line("┌", "┬", "┐"))
    print(row(headers))
    print(line("├", "┼", "┤"))
    for r in table:
        print(row(r))
    print(line("└", "┴", "┘"))

    ratios = [
        float(r["value_ratio"])
        for r in rows
        if isinstance(r["value_ratio"], (int, float)) and math.isfinite(float(r["value_ratio"]))
    ]
    if ratios:
        print(f"Best value_ratio: {max(ratios):.4f}")
        print(f"Mean value_ratio: {sum(ratios) / len(ratios):.4f}")
    else:
        print("Best value_ratio: N/A (ninguna corrida con score numérico)")
        print("Mean value_ratio: N/A")


def main() -> None:
    """Run the submit-and-poll flow for Challenge 03 (single or --repeat mode)."""
    # Fuerza UTF-8 en consolas Windows (cp1252) para los caracteres de caja y —.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Challenge 03 — Knapsack submission")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Envía el mismo prompt N veces, esperando 30s entre submissions.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        print("ERROR: --repeat debe ser >= 1.")
        sys.exit(1)

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

    if args.repeat == 1:
        submission_id, result = run_single(prompt, api_key)
        print_result(result, submission_id)
        return

    rows: list[dict] = []
    for i in range(1, args.repeat + 1):
        print(f"\n===== Corrida {i}/{args.repeat} =====")
        submission_id, result = run_single(prompt, api_key)
        metrics = result.get("metrics") or {}
        rows.append(
            {
                "n": i,
                "submission_id": submission_id,
                "value_ratio": metrics.get("value_ratio", float("nan")),
                "exact_match": metrics.get("exact_match", float("nan")),
            }
        )
        print_result(result, submission_id)
        if i < args.repeat:
            print(f"Esperando {REPEAT_DELAY}s antes de la siguiente submission...")
            time.sleep(REPEAT_DELAY)

    print("\n================ RESUMEN ================")
    print_summary_table(rows)


if __name__ == "__main__":
    main()
