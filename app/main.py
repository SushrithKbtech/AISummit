from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agent import build_agent_reply
from .callback import send_final_callback
from .config import Settings, load_settings
from .config import detect_scam_intent
from .extract import extract_intelligence, merge_extraction
from .models import ErrorResponse, IncomingRequest, ReplyResponse
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


def _safe_success(reply: str = "OK") -> JSONResponse:
    # Always return exactly {status, reply}
    return JSONResponse(
        status_code=200,
        content=ReplyResponse(status="success", reply=str(reply)).model_dump(),
    )


def _safe_error(code: int, message: str) -> JSONResponse:
    # GUVI expects schema {status, message} for errors
    return JSONResponse(
        status_code=code,
        content=ErrorResponse(status="error", message=message).model_dump(),
    )


@app.on_event("startup")
def _load_settings() -> None:
    global settings
    settings = load_settings()
    logger.info("Service started successfully")


@app.get("/")
@app.head("/")
def root_probe() -> JSONResponse:
    # Some platforms/monitors hit / or HEAD /
    return _safe_success("OK")


# IMPORTANT: GUVI tester sometimes does GET/HEAD to the endpoint URL itself
@app.api_route("/message", methods=["GET", "HEAD"])
async def message_probe() -> JSONResponse:
    return _safe_success("OK")


@app.post("/message")
async def handle_message(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> JSONResponse:
    # settings should exist after startup
    if settings is None:
        return _safe_error(500, "Server not initialized")

    # auth (keep strict)
    if not x_api_key or x_api_key != settings.api_key:
        return _safe_error(401, "Invalid API key or malformed request")

    # Read raw body ALWAYS (do not depend on FastAPI JSON parsing)
    try:
        raw = await request.body()
    except Exception:
        return _safe_success("OK")

    if not raw or raw.strip() == b"":
        # GUVI sometimes sends empty body probes
        return _safe_success("OK")

    # Try parse JSON manually (tolerate bad content-type / garbage)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return _safe_success("OK")

    if not isinstance(payload, dict):
        return _safe_success("OK")

    # Must have message.text for real processing; otherwise treat as probe
    msg = payload.get("message") if isinstance(payload.get("message"), dict) else None
    msg_text = (msg.get("text") if msg else None)
    if not isinstance(msg_text, str) or not msg_text.strip():
        return _safe_success("OK")

    # Try strict pydantic validation; if fails -> safe fallback reply (NOT error)
    incoming: Optional[IncomingRequest] = None
    try:
        incoming = IncomingRequest.model_validate(payload)
    except Exception:
        session_id_raw = payload.get("sessionId")
        sender = msg.get("sender") if msg else None
        text = msg.get("text") if msg else None
        timestamp = msg.get("timestamp") if msg else None

        if not isinstance(timestamp, str) or not timestamp.strip():
            timestamp = datetime.now(timezone.utc).isoformat()

        if isinstance(session_id_raw, str) and isinstance(sender, str) and isinstance(text, str):
            sanitized = {
                "sessionId": session_id_raw,
                "message": {"sender": sender, "text": text, "timestamp": timestamp},
                "conversationHistory": payload.get("conversationHistory") if isinstance(payload.get("conversationHistory"), list) else [],
                "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
            }
            try:
                incoming = IncomingRequest.model_validate(sanitized)
            except Exception:
                incoming = None

        if incoming is None:
            fallback_options = [
                "I'm a bit confused. Which department is this and what's your employee ID?",
                "Can you share the official helpline number and a reference ID?",
                "I can't verify this right now. Do you have an official link or ticket number?",
                "Please share your branch/department and a callback number.",
            ]
            if isinstance(session_id_raw, str) and session_id_raw.strip():
                sid = session_id_raw.strip()
                state = store.get(sid)
                if state is None:
                    state = store.initialize(sid)
                state.totalMessagesExchanged += 1
                idx = state.totalMessagesExchanged % len(fallback_options)
                reply = fallback_options[idx]
                if reply == state.lastReply:
                    reply = fallback_options[(idx + 1) % len(fallback_options)]
                state.lastReply = reply
                store.upsert(state)
                return _safe_success(reply)

            # last resort deterministic selection
            idx = abs(hash(msg_text)) % len(fallback_options)
            return _safe_success(fallback_options[idx])

    # At this point incoming is valid
    session_id = incoming.sessionId.strip()
    if not session_id:
        return _safe_success("OK")

    state = store.get(session_id)
    if state is None:
        state = store.initialize(session_id)

    incoming_text = incoming.message.text or ""
    is_scammer = incoming.message.sender == "scammer"

    # increment once per request only
    state.totalMessagesExchanged += 1

    # Maintain rolling memory (this was missing before)
    if is_scammer:
        state.recentScammer = (state.recentScammer + [incoming_text])[-3:]
    else:
        state.recentHoneypot = (state.recentHoneypot + [incoming_text])[-3:]

    if is_scammer:
        detector = detect_scam_intent(incoming_text)
        logger.info(
            "scam_detector score=%s indicators=%s threshold=%s",
            detector.score,
            detector.indicators,
            settings.scam_threshold,
        )
        state.scamScore = max(state.scamScore, detector.score)

        if state.scamScore >= settings.scam_threshold:
            state.scamConfirmed = True
            state.agentActive = True
        elif any(k in incoming_text.lower() for k in ["otp", "upi", "blocked", "verify", "kyc"]):
            state.scamConfirmed = True
            state.agentActive = True

        extraction = extract_intelligence(incoming_text)
        state.extractedIntelligence = merge_extraction(state.extractedIntelligence, extraction)

        # track missing slots
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

        normalized = incoming_text.strip().lower()
        if state.lastScammerMessage and normalized == state.lastScammerMessage:
            state.turnsSinceChange += 1
        else:
            state.turnsSinceChange = 0
            state.lastScammerMessage = normalized

    # build reply
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

    # avoid duplicate replies
    if reply_text == state.lastReply:
        reply_text = "I need a moment to check."
        if reply_text == state.lastReply:
            reply_text = "Sorry, I still don't understand."

    state.lastReply = reply_text

    # termination logic
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

    # final callback
    if state.terminated and state.scamConfirmed and not state.finalCallbackSent:
        sent = send_final_callback(state, settings)
        state.finalCallbackSent = True
        if not sent:
            logger.warning("Final callback failed for session %s", session_id)

    store.upsert(state)
    return _safe_success(reply_text)
