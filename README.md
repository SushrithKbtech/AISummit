# Agentic Honeypot API

Production-ready API for scam detection, autonomous engagement, intelligence extraction, and mandatory final callback.

## Requirements
- Python 3.10+

## Setup
```bash
pip install -r requirements.txt
```

## Environment variables
- `HONEY_POT_API_KEY` (required)
- `SCAM_THRESHOLD` (default: 0.6)
- `MAX_TURNS` (default: 10)
- `FINAL_CALLBACK_URL` (default: https://hackathon.guvi.in/api/updateHoneyPotFinalResult)
- `HTTP_TIMEOUT_SECONDS` (default: 5)
- `PERSONA_NAME` (default: Sam)

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

To serve HTTPS directly, pass `--ssl-keyfile` and `--ssl-certfile` to uvicorn or place the service behind a TLS-terminating proxy.
