from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

from .models import AgentReply, Message, SessionState

# Bounded-action design: the LLM can only choose among fixed strategies,
# and then rephrase a reply. Deterministic templates are the source of truth.

STRATEGIES = [
    "ASK_EMPLOYEE_ID_BRANCH",
    "ASK_OFFICIAL_LINK_TICKET",
    "CALL_BACK_CONFIRM_NUMBER",
    "UPI_COLLECT_REQUEST_CHECK",
    "TECHNICAL_STALL_APP_ISSUE",
    "TECHNICAL_STALL_PIN_ISSUE",
]

_SLOT_STRATEGY = {
    "upi": "UPI_COLLECT_REQUEST_CHECK",
    "phone": "CALL_BACK_CONFIRM_NUMBER",
    "phishing": "ASK_OFFICIAL_LINK_TICKET",
    "bank": "ASK_EMPLOYEE_ID_BRANCH",
    "suspicious": "ASK_OFFICIAL_LINK_TICKET",
}

_STRATEGY_TEMPLATES: Dict[str, List[str]] = {
    "ASK_EMPLOYEE_ID_BRANCH": [
        "Sorry, who is this exactly? What's your name and employee ID?",
        "Which team/branch is this from? Please share your name and ID.",
        "Can you share your name and employee ID so I can verify?",
        "Which branch are you calling from? Name and ID please.",
    ],
    "ASK_OFFICIAL_LINK_TICKET": [
        "Do you have an official link or reference/ticket number?",
        "Can you share a ticket number or an official link I can check?",
        "Is there any official reference ID for this case?",
        "Please send the official link or case reference number.",
    ],
    "CALL_BACK_CONFIRM_NUMBER": [
        "What's the official helpline number? I want to call back and verify.",
        "Can I call the official helpline? Share the number please.",
        "Please give the official customer care number. I'll call back.",
        "Which number should I call to verify this officially?",
    ],
    "UPI_COLLECT_REQUEST_CHECK": [
        "If it's a collect request, what's the UPI handle it's coming from?",
        "Which UPI handle is raising the request? I want to verify in the app.",
        "Please share the UPI handle so I can check the collect request.",
        "What UPI ID is the request from? I can verify on my app.",
    ],
    "TECHNICAL_STALL_APP_ISSUE": [
        "My app is acting up. Can you share details and call back in a bit?",
        "I'm in a meeting—can you send the details and call later?",
        "My phone is restarting. Can you wait a few minutes?",
        "Network is bad right now. Can you share the official link/ticket?",
    ],
    # This is the “friend-like” human reply style (PIN screen issue) ✅
    "TECHNICAL_STALL_PIN_ISSUE": [
        "Sir, I'm a bit confused. The app is asking for my PIN but it's not working—what should I do? Also what's the official helpline number or reference ID?",
        "I'm confused—my app keeps asking for my PIN and then errors out. Can you tell me the steps? Also share an official link or ticket number.",
        "My banking app is stuck on the PIN screen and won't proceed. What are the steps to fix it? What's your official helpline number?",
        "The app is asking for my PIN but it's failing. Can you guide me what to do next? Please share the official reference ID.",
    ],
}

# IMPORTANT:
# - Mentioning "PIN screen" is allowed.
# - Asking the user to SHARE OTP/PIN/account details is NOT allowed.
_BANNED_PATTERNS = [
    "share otp",
    "send otp",
    "tell me otp",
    "otp here",
    "one time password",
    "one-time password",
    "password",
    "cvv",
    "account number",
    "debit card",
    "credit card",
    "netbanking password",
]

_SECRET_ASK_PATTERNS = [
    "share your pin",
    "tell me your pin",
    "send your pin",
    "what is your pin",
    "enter your pin here",
    "share your otp",
    "tell me the otp",
    "send otp here",
    "share your account number",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _limit_sentences(text: str, max_sentences: int = 2) -> str:
    # FIXED: your old regex had "\\s+" which matched literal "\s" not whitespace
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:max_sentences]).strip()


def _extract_last_messages(history: List[Message]) -> Tuple[List[str], List[str]]:
    scammer_msgs = [msg.text for msg in history if msg.sender == "scammer"]
    user_msgs = [msg.text for msg in history if msg.sender == "user"]
    return scammer_msgs[-3:], user_msgs[-3:]


def _pick_from_templates(strategy: str, state: SessionState) -> str:
    options = _STRATEGY_TEMPLATES.get(strategy, ["I need a moment to check."])
    seed = f"{state.sessionId}:{state.totalMessagesExchanged}:{strategy}"
    index = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(options)
    reply = options[index]
    if reply == state.lastReply:
        reply = options[(index + 1) % len(options)]
    return reply


def _pick_deterministic_strategy(state: SessionState, scammer_text: str) -> str:
    t = (scammer_text or "").lower()

    # If scammer pushes OTP/PIN/password -> stall on PIN screen issue + ask official details
    if any(k in t for k in ["otp", "pin", "password", "passcode"]):
        return "TECHNICAL_STALL_PIN_ISSUE"

    # If scammer mentions UPI/collect -> ask UPI handle
    if any(k in t for k in ["upi", "collect", "request"]):
        return "UPI_COLLECT_REQUEST_CHECK"

    # Otherwise pick based on missing slots
    for slot in getattr(state, "missingSlots", []):
        strategy = _SLOT_STRATEGY.get(slot)
        if strategy:
            return strategy

    # fallback rotation
    index = state.totalMessagesExchanged % len(STRATEGIES)
    return STRATEGIES[index]


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _llm_select_strategy(
    state: SessionState,
    scammer_text: str,
    history: List[Message],
    api_key: str,
    model: str,
) -> Optional[str]:
    if OpenAI is None or not api_key:
        return None

    scammer_msgs, user_msgs = _extract_last_messages(history)
    missing = {
        "upi": "upi" in state.missingSlots,
        "phone": "phone" in state.missingSlots,
        "phishing": "phishing" in state.missingSlots,
        "bank": "bank" in state.missingSlots,
        "suspicious": "suspicious" in state.missingSlots,
    }
    payload = {
        "lastScammerMessages": scammer_msgs,
        "lastHoneypotReplies": user_msgs,
        "extractedIntelligence": state.extractedIntelligence.model_dump(),
        "missingSlots": missing,
        "enum": STRATEGIES,
        "scammerMessage": scammer_text,
    }

    system_prompt = (
        "Return JSON only. Select the safest next strategy from the enum.\n"
        "Prefer strategies that extract missing intel: official link/ticket, helpline number, employee ID/branch, UPI handle.\n"
        "If scammer demands OTP/PIN/password, choose TECHNICAL_STALL_PIN_ISSUE.\n"
        "Never choose any strategy that asks for victim secrets.\n"
        "JSON schema: {\"strategy\": <enum>, \"reason\": <short>}"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content if response.choices else ""
        data = _parse_json(content or "")
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    strategy = data.get("strategy")
    if strategy in STRATEGIES:
        return strategy
    return None


def _llm_generate_reply(
    strategy: str,
    scammer_text: str,
    api_key: str,
    model: str,
    recent_scammer: Optional[List[str]] = None,
    recent_honeypot: Optional[List[str]] = None,
    safety_note: str = "",
    early_turn: bool = False,
) -> Optional[str]:
    if OpenAI is None or not api_key:
        return None

    recent_scammer = (recent_scammer or [])[-3:]
    recent_honeypot = (recent_honeypot or [])[-3:]

    softener = (
        "Be extra polite and non-confrontational; ask what this is about."
        if early_turn
        else "Stay calm, slightly worried, and cautious."
    )

    system_prompt = (
        "You are a real person texting back to a scammer. "
        "Sound natural, slightly confused and worried. "
        "Reply in 1-2 short sentences only. "
        "Based on scammerMessage + recent context, ask ONE best follow-up that extracts intel. "
        "Try to get one of: official link/ticket, callback/helpline number, employee ID/branch, UPI handle. "
        "Never ask for OTP, PIN, passwords, account number, CVV, or any victim credential. "
        "Allowed: mention the app is asking for PIN and it's not working, but DO NOT request the PIN. "
        "Do not repeat the same question as recentHoneypot; rephrase or ask a different missing detail. "
        "Never accuse them of being a scammer or mention scam detection. "
        f"{softener} Output JSON only: {{\"reply\": string}}. {safety_note}"
    )

    user_prompt = json.dumps(
        {
            "strategy": strategy,
            "scammerMessage": scammer_text,
            "recentScammer": recent_scammer,
            "recentHoneypot": recent_honeypot,
        }
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
        )
        content = response.choices[0].message.content if response.choices else ""
        data = _parse_json(content or "")
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    reply = data.get("reply")
    if isinstance(reply, str):
        return reply.strip() or None
    return None


def _contains_banned(text: str) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in _BANNED_PATTERNS)


def _asks_for_secret(text: str) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in _SECRET_ASK_PATTERNS)


def _asks_for_details(text: str) -> bool:
    lowered = (text or "").lower()
    keywords = [
        "employee",
        "department",
        "branch",
        "official",
        "link",
        "ticket",
        "reference",
        "helpline",
        "number",
        "upi",
        "handle",
    ]
    return "?" in lowered and any(keyword in lowered for keyword in keywords)


def build_agent_reply(
    state: SessionState,
    scammer_text: str,
    history: List[Message],
    api_key: str,
    model: str,
) -> AgentReply:
    # Choose strategy
    strategy = _llm_select_strategy(state, scammer_text, history, api_key, model)
    if not strategy:
        strategy = _pick_deterministic_strategy(state, scammer_text)

    early_turn = state.totalMessagesExchanged <= 3

    # Use rolling memory if present in state (works even if GUVI doesn't send conversationHistory)
    recent_scammer = getattr(state, "recentScammer", [])
    recent_honeypot = getattr(state, "recentHoneypot", [])

    reply = _llm_generate_reply(
        strategy,
        scammer_text,
        api_key,
        model,
        recent_scammer=recent_scammer,
        recent_honeypot=recent_honeypot,
        early_turn=early_turn,
    )
    if not reply:
        reply = _pick_from_templates(strategy, state)

    reply = _limit_sentences(reply, max_sentences=2)

    # Safety enforcement: never ask secrets
    if _asks_for_secret(reply) or _contains_banned(reply):
        reply = _llm_generate_reply(
            strategy,
            scammer_text,
            api_key,
            model,
            recent_scammer=recent_scammer,
            recent_honeypot=recent_honeypot,
            safety_note="Do not ask for OTP/PIN/password/account details. Ask only for official verification info.",
            early_turn=early_turn,
        ) or _pick_from_templates("ASK_OFFICIAL_LINK_TICKET", state)

    reply = _limit_sentences(reply, max_sentences=2)

    # Avoid repetition against last reply
    if _normalize_text(reply) == _normalize_text(state.lastReply or ""):
        alt_strategy = STRATEGIES[(STRATEGIES.index(strategy) + 1) % len(STRATEGIES)]
        alt_reply = _llm_generate_reply(
            alt_strategy,
            scammer_text,
            api_key,
            model,
            recent_scammer=recent_scammer,
            recent_honeypot=recent_honeypot,
            early_turn=early_turn,
        )
        reply = alt_reply or _pick_from_templates(alt_strategy, state)

    reply = _limit_sentences(reply, max_sentences=2)

    # Ensure it actually asks for scammer-side verification details
    if not _asks_for_details(reply):
        reply = _pick_from_templates("ASK_OFFICIAL_LINK_TICKET", state)

    # Final guardrail
    if _asks_for_secret(reply) or _contains_banned(reply):
        reply = "Sorry, I can't share any codes. Can you send the official link or reference number?"

    agent_notes = (
        f"Strategy: {strategy}. "
        "Goal: keep scammer engaged and extract verification details (link/ticket, helpline, UPI handle, ID)."
    )

    should_terminate = bool(getattr(state, "terminated", False))
    return AgentReply(reply=reply, agentNotes=agent_notes, shouldTerminate=should_terminate)
