## Environment variables
- `HONEY_POT_API_KEY` (required)
- `OPENAI_API_KEY` (optional, enables LLM replies)
- `OPENAI_MODEL` (default: gpt-4o-mini)
- `SCAM_THRESHOLD` (default: 0.5)
- `MAX_TURNS` (default: 20)
- `CALLBACK_TIMEOUT` (default: 5)
- `FINAL_CALLBACK_URL` (default: https://hackathon.guvi.in/api/updateHoneyPotFinalResult)

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
