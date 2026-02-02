from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

from .models import Intelligence, SessionState


class SessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: Dict[str, SessionState] = {}

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return state.model_copy(deep=True)

    def upsert(self, state: SessionState) -> None:
        with self._lock:
            self._sessions[state.sessionId] = state

    def initialize(self, session_id: str) -> SessionState:
        with self._lock:
            state = SessionState(
                sessionId=session_id,
                extractedIntelligence=Intelligence(
                    bankAccounts=[],
                    upiIds=[],
                    phishingLinks=[],
                    phoneNumbers=[],
                    suspiciousKeywords=[],
                ),
                missingSlots=["upi", "phone", "phishing", "bank", "suspicious"],
                recentScammer=[],
                recentHoneypot=[],
            )
            self._sessions[session_id] = state
            return state.model_copy(deep=True)
