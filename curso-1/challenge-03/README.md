# Challenge 03 — Knapsack Logistics Optimizer

Prompt engineering challenge: given a drone with limited capacity and a list of packages, produce an ordered priority list that maximises total loaded value using a Claude-powered knapsack solver.

## Setup

```bash
# Install dependencies
uv sync

# Configure credentials
cp .env.example .env
# Edit .env and set DSRP_API_KEY=<your-key>
```

## Run

```bash
uv run python solution.py
```

## Modify the prompt

Edit `prompt.txt` to tune the system prompt, then re-run `solution.py`. The script validates the prompt is under 8 000 characters before submitting.

## Target

`value_ratio > 0.9987` (beat current leaderboard #1 — student-01)
