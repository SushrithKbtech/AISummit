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
- `MAX_TURNS` (default: 18)
- `FINAL_CALLBACK_URL` (default: https://hackathon.guvi.in/api/updateHoneyPotFinalResult)
- `CALLBACK_TIMEOUT` (default: 5)
- `PERSONA_NAME` (default: Sam)

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

To serve HTTPS directly, pass `--ssl-keyfile` and `--ssl-certfile` to uvicorn or place the service behind a TLS-terminating proxy.

## Example curl
First message:
```bash
curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_SECRET_API_KEY" \
  -d "{\"sessionId\":\"abc123\",\"message\":{\"sender\":\"scammer\",\"text\":\"Your account is blocked. Verify now.\",\"timestamp\":\"2026-02-02T10:00:00Z\"},\"conversationHistory\":[],\"metadata\":{\"channel\":\"SMS\",\"language\":\"English\",\"locale\":\"IN\"}}"
```

Follow-up message:
```bash
curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_SECRET_API_KEY" \
  -d "{\"sessionId\":\"abc123\",\"message\":{\"sender\":\"scammer\",\"text\":\"Share your OTP\",\"timestamp\":\"2026-02-02T10:01:00Z\"},\"conversationHistory\":[{\"sender\":\"scammer\",\"text\":\"Your account is blocked. Verify now.\",\"timestamp\":\"2026-02-02T10:00:00Z\"}],\"metadata\":{\"channel\":\"SMS\",\"language\":\"English\",\"locale\":\"IN\"}}"
```
