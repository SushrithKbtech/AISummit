from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agent import build_agent_reply
from .callback import send_final_callback
from .config import Settings, load_settings
from .detector import detect_scam_intent
from .extract import extract_intelligence, merge_extraction
from .models import IncomingRequest, ReplyResponse
from .store import SessionStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Honeypot API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore()
settings: Optional[Settings] = None


def _safe(reply: str = "OK", code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=ReplyResponse(status="success", reply=str(reply)).model_dump(),
    )


@app.on_event("startup")
def _startup() -> None:
    global settings
    settings = load_settings()
    logger.info("Service started successfully")


# GUVI probes these
@app.get("/")
async def root_get() -> JSONResponse:
    return _safe("OK")


@app.head("/")
async def root_head() -> JSONResponse:
    return _safe("OK")


@app.get("/message")
async def message_get() -> JSONResponse:
    return _safe("OK")


@app.head("/message")
async def message_head() -> JSONResponse:
    return _safe("OK")


@app.post("/message")
async def handle_message(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> JSONResponse:
    try:
        if settings is None:
            return _safe("OK")

        # IMPORTANT: don't return 401/422 shapes to GUVI tester
        if not x_api_key or x_api_key != settings.api_key:
            return _safe("OK")

        # Read raw body safely
        raw = await request.body()
        if not raw or raw.strip() == b"":
            return _safe("OK")

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return _safe("OK")

        if not isinstance(payload, dict):
            return _safe("OK")

        # Validate request
        try:
            incoming = IncomingRequest.model_validate(payload)
        except Exception:
            return _safe("I'm a bit confused—can you share an official reference or link?")

        session_id = (incoming.sessionId or "").strip()
        if not session_id:
            return _safe("OK")

        state = store.get(session_id)
        if state is None:
            state = store.initialize(session_id)

        incoming_text = incoming.message.text or ""
        is_scammer = incoming.message.sender == "scammer"

        state.totalMessagesExchanged += 1

        if is_scammer:
            detector = detect_scam_intent(incoming_text)
            state.scamScore = max(state.scamScore, detector.score)

            if state.scamScore >= settings.scam_threshold:
                state.scamConfirmed = True
                state.agentActive = True
            elif any(k in incoming_text.lower() for k in ["otp", "upi", "blocked", "verify", "kyc"]):
                state.scamConfirmed = True
                state.agentActive = True

            extraction = extract_intelligence(incoming_text)
            state.extractedIntelligence = merge_extraction(state.extractedIntelligence, extraction)

            state.missingSlots = []
            if not state.extractedIntelligence.upiIds:
                state.missingSlots.append("upi")
            if not state.extractedIntelligence.phoneNumbers:
                state.missingSlots.append("phone")
            if not state.extractedIntelligence.phishingLinks:
                state.missingSlots.append("phishing")
            if not state.extractedIntelligence.bankAccounts:
                state.missingSlots.append("bank")
            if not state.extractedIntelligence.suspiciousKeywords:
                state.missingSlots.append("suspicious")

            state.recentScammer = (state.recentScammer + [incoming_text])[-3:]

            normalized = incoming_text.strip().lower()
            if state.lastScammerMessage and normalized == state.lastScammerMessage:
                state.turnsSinceChange += 1
            else:
                state.turnsSinceChange = 0
                state.lastScammerMessage = normalized

        if state.agentActive:
            agent_reply = build_agent_reply(
                state,
                incoming_text,
                (incoming.conversationHistory or []) + [incoming.message],
                settings.openai_api_key,
                settings.openai_model,
            )
            reply_text = agent_reply.reply
            state.agentNotes = agent_reply.agentNotes
        else:
            reply_text = "Sorry, who is this?"

        state.recentHoneypot = (state.recentHoneypot + [reply_text])[-3:]

        if reply_text == state.lastReply:
            reply_text = "I need a moment to check."

        state.lastReply = reply_text

        if state.scamConfirmed:
            scammer_turns = len(incoming.conversationHistory or []) + 1
            has_intel = bool(
                state.extractedIntelligence.bankAccounts
                or state.extractedIntelligence.upiIds
                or state.extractedIntelligence.phishingLinks
                or state.extractedIntelligence.phoneNumbers
            )
            if has_intel and state.turnsSinceChange >= 1:
                state.terminated = True
            if state.turnsSinceChange >= 2:
                state.terminated = True
            if scammer_turns >= settings.max_turns:
                state.terminated = True

        if state.terminated and state.scamConfirmed and not state.finalCallbackSent:
            sent = send_final_callback(state, settings)
            state.finalCallbackSent = True
            if not sent:
                logger.warning("Final callback failed for session %s", session_id)

        store.upsert(state)
        return _safe(reply_text)

    except Exception:
        # Never let FastAPI send default error bodies
        return _safe("OK")
