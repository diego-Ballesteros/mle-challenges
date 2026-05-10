"""Challenge 01 — Hola Mundo: authenticated POST to the DSRP course API."""

import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

ENDPOINT = "https://utpyawapnk.execute-api.us-east-1.amazonaws.com/Prod/challenges/01/hello"

PROFILE: dict = {
    "name": "Diego Andres Ballesteros Cediel",
    "occupation": "Desarrollador de Software Junior | Transitioning to ML Engineer",
    "interests": [
        "machine learning",
        "mlops",
        "python",
        "backend development",
        "software architecture",
    ],
    "bio": (
        "Vengo del mundo del desarrollo de software y estoy construyendo mi camino hacia "
        "Machine Learning Engineering. Apasionado por llevar modelos a producción con "
        "buenas prácticas de ingeniería."
    ),
    "github_url": "https://github.com/diego-Ballesteros",
    "linkedin_url": "https://www.linkedin.com/in/diego-ballesteros-ac/",
}


def load_api_key() -> str:
    """Load DSRP_API_KEY from .env file.

    Returns:
        The API key string.

    Raises:
        SystemExit: If the key is missing or empty.
    """
    load_dotenv()
    key = os.getenv("DSRP_API_KEY", "").strip()
    if not key:
        print("API key inválida o faltante")
        raise SystemExit(1)
    return key


def print_header(timestamp: str) -> None:
    """Print the challenge header banner.

    Args:
        timestamp: ISO-formatted datetime string for the run.
    """
    print("=" * 52)
    print("  DSRP — Challenge 01: Hola Mundo")
    print("=" * 52)
    print(f"  Endpoint : {ENDPOINT}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 52)


def print_result(response_data: dict) -> None:
    """Print the submission result in a formatted block.

    Args:
        response_data: Parsed JSON response from the API.
    """
    submission_id = response_data.get("submission_id", "N/A")
    timestamp = response_data.get("timestamp", "N/A")

    print("\n" + "─" * 52)
    print(f"  submission_id : {submission_id}")
    print(f"  timestamp     : {timestamp}")
    print("─" * 52)
    print("  Guarda tu submission_id")


def submit_profile(api_key: str) -> None:
    """Send the profile payload to the challenge endpoint.

    Handles HTTP 401, 422, timeout, and connection errors explicitly.

    Args:
        api_key: DSRP API key used in the x-api-key header.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_header(timestamp)

    print("\n📤 Enviando perfil...")
    print(json.dumps(PROFILE, indent=2, ensure_ascii=False))

    try:
        response = requests.post(
            ENDPOINT,
            headers={"x-api-key": api_key},
            json=PROFILE,
            timeout=15,
        )
    except requests.exceptions.Timeout:
        print("Timeout — servidor no respondió")
        raise SystemExit(1)
    except requests.exceptions.ConnectionError:
        print("No se pudo conectar al servidor")
        raise SystemExit(1)

    if response.status_code == 401:
        print("API key inválida o faltante")
        raise SystemExit(1)

    if response.status_code == 422:
        detail = response.json().get("detail", response.text)
        print(f"Error de validación: {detail}")
        raise SystemExit(1)

    response.raise_for_status()

    print("\n✅ Submission exitosa!")
    print_result(response.json())


def main() -> None:
    """Entry point: load credentials and submit the challenge profile."""
    api_key = load_api_key()
    submit_profile(api_key)


if __name__ == "__main__":
    main()
