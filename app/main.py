from __future__ import annotations

import logging
from typing import Optional

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .agent import build_agent_reply
from .callback import send_final_callback
from .config import Settings, load_settings
from .detector import detect_scam_intent
from .extract import extract_intelligence, merge_extraction
from .models import ErrorResponse, IncomingRequest, ReplyResponse
from .store import SessionStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic Honeypot API", docs_url=None, redoc_url=None)
store = SessionStore()
settings: Optional[Settings] = None


@app.on_event("startup")
def _load_settings() -> None:
    global settings
    settings = load_settings()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(status="error", error=str(exc.detail)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content=ErrorResponse(status="error", error="Invalid request payload").model_dump())


@app.post("/message")
async def handle_message(
    request: Request,
    payload: Optional[dict] = Body(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> JSONResponse:
    if settings is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if payload is None or payload == {} or "sessionId" not in payload:
        return JSONResponse(status_code=200, content=ReplyResponse(status="success", reply="Hello").model_dump())

    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")

    try:
        incoming = IncomingRequest.model_validate(payload)
    except Exception:
        return JSONResponse(status_code=200, content=ReplyResponse(status="success", reply="Hello").model_dump())

    session_id = incoming.sessionId.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId is required")

    state = store.get(session_id)
    if state is None:
        state = store.initialize(session_id)
    else:
        if incoming.conversationHistory is None or len(incoming.conversationHistory) == 0:
            raise HTTPException(status_code=400, detail="conversationHistory required for follow-up messages")

    incoming_text = incoming.message.text or ""
    is_scammer = incoming.message.sender == "scammer"

    state.totalMessages += 1

    if is_scammer:
        detector = detect_scam_intent(incoming_text)
        state.scamScore = max(state.scamScore, detector.score)
        if state.scamScore >= settings.scam_threshold:
            state.scamConfirmed = True
            state.agentActive = True

        extraction = extract_intelligence(incoming_text)
        merged = merge_extraction(state.extractedIntelligence, extraction)
        state.extractedIntelligence = merged

        normalized = incoming_text.strip().lower()
        if state.lastScammerMessage and normalized == state.lastScammerMessage:
            state.turnsSinceChange += 1
        else:
            state.turnsSinceChange = 0
            state.lastScammerMessage = normalized

    reply_text = ""
    if state.agentActive:
        agent_reply = build_agent_reply(state, incoming_text)
        reply_text = agent_reply.reply
    else:
        reply_text = "Sorry, who is this?"

    if reply_text == state.lastReply:
        reply_text = "I need a moment to check."
        if reply_text == state.lastReply:
            reply_text = "Sorry, I still don't understand."

    state.lastReply = reply_text
    state.totalMessages += 1

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
    return JSONResponse(status_code=200, content=ReplyResponse(status="success", reply=reply_text).model_dump())
