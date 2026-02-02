from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_key: str
    scam_threshold: float
    max_turns: int
    callback_url: str
    http_timeout_seconds: float
    persona_name: str


def load_settings() -> Settings:
    api_key = os.environ.get("HONEY_POT_API_KEY", "")
    if not api_key:
        raise RuntimeError("HONEY_POT_API_KEY is required")

    scam_threshold = float(os.environ.get("SCAM_THRESHOLD", "0.6"))
    max_turns = int(os.environ.get("MAX_TURNS", "20"))
    callback_url = os.environ.get(
        "FINAL_CALLBACK_URL",
        "https://hackathon.guvi.in/api/updateHoneyPotFinalResult",
    )
    http_timeout_seconds = float(os.environ.get("CALLBACK_TIMEOUT", os.environ.get("HTTP_TIMEOUT_SECONDS", "5")))
    persona_name = os.environ.get("PERSONA_NAME", "Sam")

    return Settings(
        api_key=api_key,
        scam_threshold=scam_threshold,
        max_turns=max_turns,
        callback_url=callback_url,
        http_timeout_seconds=http_timeout_seconds,
        persona_name=persona_name,
    )
