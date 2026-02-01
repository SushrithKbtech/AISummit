from __future__ import annotations

import re
from typing import List

from .models import DetectorResult


_SCAM_KEYWORDS = {
    "urgent": 0.15,
    "immediately": 0.12,
    "verify": 0.12,
    "verification": 0.1,
    "account": 0.08,
    "suspended": 0.12,
    "blocked": 0.12,
    "kyc": 0.12,
    "otp": 0.1,
    "password": 0.1,
    "bank": 0.08,
    "police": 0.2,
    "court": 0.2,
    "fine": 0.15,
    "penalty": 0.12,
    "payment": 0.08,
    "transfer": 0.1,
    "upi": 0.12,
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
