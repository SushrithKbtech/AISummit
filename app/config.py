from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from .models import DetectorResult

_SCAM_KEYWORDS = {
    "urgent": 0.22,
    "immediately": 0.15,
    "verify": 0.18,
    "verification": 0.12,
    "account blocked": 0.22,
    "account": 0.08,
    "suspended": 0.16,
    "blocked": 0.16,
    "kyc": 0.18,
    "otp": 0.2,
    "password": 0.12,
    "bank": 0.08,
    "fraud prevention team": 0.22,
    "customer care": 0.18,
    "police": 0.2,
    "court": 0.2,
    "fine": 0.15,
    "penalty": 0.12,
    "payment": 0.08,
    "transfer": 0.1,
    "upi": 0.18,
    "gift card": 0.18,
    "bitcoin": 0.2,
    "crypto": 0.15,
    "wire": 0.1,
}

_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s\-]{7,}\d)")


def detect_scam_intent(text: str) -> DetectorResult:
    normalized = (text or "").lower()
    indicators: List[str] = []
    score = 0.0

    for keyword, weight in _SCAM_KEYWORDS.items():
        if keyword in normalized:
            indicators.append(keyword)
            score += weight

    if _URL_PATTERN.search(normalized):
        indicators.append("url")
        score += 0.2

    if _PHONE_PATTERN.search(normalized):
        indicators.append("phone")
        score += 0.1

    score = min(score, 1.0)
    return DetectorResult(score=score, indicators=sorted(set(indicators)))


@dataclass(frozen=True)
class Settings:
    api_key: str
    scam_threshold: float
    max_turns: int
    callback_url: str
    http_timeout_seconds: float
    persona_name: str
    openai_api_key: str
    openai_model: str


def load_settings() -> Settings:
    if load_dotenv is not None:
        load_dotenv()

    api_key = os.environ.get("HONEY_POT_API_KEY", "")
    if not api_key:
        raise RuntimeError("HONEY_POT_API_KEY is required")

    scam_threshold = float(os.environ.get("SCAM_THRESHOLD", "0.5"))
    max_turns = int(os.environ.get("MAX_TURNS", "20"))

    callback_url = os.environ.get(
        "FINAL_CALLBACK_URL",
        "https://hackathon.guvi.in/api/updateHoneyPotFinalResult",
    )

    http_timeout_seconds = float(os.environ.get("CALLBACK_TIMEOUT", os.environ.get("HTTP_TIMEOUT_SECONDS", "5")))
    persona_name = os.environ.get("PERSONA_NAME", "Sam")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    return Settings(
        api_key=api_key,
        scam_threshold=scam_threshold,
        max_turns=max_turns,
        callback_url=callback_url,
        http_timeout_seconds=http_timeout_seconds,
        persona_name=persona_name,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )
