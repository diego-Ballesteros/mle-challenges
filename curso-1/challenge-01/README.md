# Challenge 01 — Hola Mundo

First authenticated POST request to the DSRP course API. Submits a personal profile payload and receives a `submission_id` on success.

## Setup

1. Navigate into the challenge directory:
   ```bash
   cd curso-1/challenge-01
   ```

2. Install dependencies with UV:
   ```bash
   uv sync
   ```

3. Copy the example env file and fill in your API key:
   ```bash
   cp .env.example .env
   # then edit .env and set DSRP_API_KEY=<your-key>
   ```

## Run

```bash
uv run python solution.py
```

## Expected Output

```
====================================================
  DSRP — Challenge 01: Hola Mundo
====================================================
  Endpoint : https://utpyawapnk.execute-api.us-east-1.amazonaws.com/Prod/challenges/01/hello
  Timestamp: 2026-xx-xx xx:xx:xx
====================================================
📤 Enviando perfil...
{...json body...}
✅ Submission exitosa!

────────────────────────────────────────────────────
  submission_id : xxxx-xxxx-xxxx
  timestamp     : xxxx
────────────────────────────────────────────────────
  Guarda tu submission_id
```
