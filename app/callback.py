from __future__ import annotations

import logging
import requests

from .config import Settings
from .models import FinalCallbackPayload, SessionState

logger = logging.getLogger(__name__)


def send_final_callback(state: SessionState, settings: Settings) -> bool:
    payload = FinalCallbackPayload(
        sessionId=state.sessionId,
        scamDetected=bool(state.scamConfirmed),  # ✅ honest
        totalMessagesExchanged=state.totalMessagesExchanged,
        extractedIntelligence=state.extractedIntelligence,
        agentNotes=state.agentNotes or _build_agent_notes(state),
    )

    try:
        response = requests.post(
            settings.callback_url,
            json=payload.model_dump(),
            headers={"Content-Type": "application/json"},
            timeout=settings.http_timeout_seconds,
        )
        if 200 <= response.status_code < 300:
            return True
        logger.warning("Final callback failed with status %s", response.status_code)
    except requests.RequestException:
        logger.exception("Final callback request failed")

    return False


def _build_agent_notes(state: SessionState) -> str:
    details = []
    if state.extractedIntelligence.phishingLinks:
        details.append("Phishing link provided")
    if state.extractedIntelligence.upiIds or state.extractedIntelligence.bankAccounts:
        details.append("Payment details requested")
    if state.extractedIntelligence.phoneNumbers:
        details.append("Phone number shared")
    if state.extractedIntelligence.suspiciousKeywords:
        details.append("Suspicious language used")

    if not details:
        details.append("Scammer engaged with pressure tactics")

    return "; ".join(details)
