from __future__ import annotations

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

_STRATEGY_INTENT = {
    "ASK_EMPLOYEE_ID_BRANCH": "clarify_procedure",
    "ASK_OFFICIAL_LINK_TICKET": "clarify_procedure",
    "CALL_BACK_CONFIRM_NUMBER": "clarify_procedure",
    "UPI_COLLECT_REQUEST_CHECK": "clarify_procedure",
    "TECHNICAL_STALL_APP_ISSUE": "pretend_technical_issue",
    "TECHNICAL_STALL_PIN_ISSUE": "pretend_technical_issue",
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
    "passcode",
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
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:max_sentences]).strip()


def _extract_last_messages(history: List[Message]) -> Tuple[List[str], List[str]]:
    scammer_msgs = [msg.text for msg in history if msg.sender == "scammer"]
    user_msgs = [msg.text for msg in history if msg.sender == "user"]
    return scammer_msgs[-3:], user_msgs[-3:]


def _pick_deterministic_strategy(state: SessionState, scammer_text: str) -> str:
    t = (scammer_text or "").lower()

    if any(k in t for k in ["otp", "pin", "password", "passcode"]):
        return "TECHNICAL_STALL_PIN_ISSUE"

    if any(k in t for k in ["upi", "collect", "request"]):
        return "UPI_COLLECT_REQUEST_CHECK"

    for slot in getattr(state, "missingSlots", []):
        strategy = _SLOT_STRATEGY.get(slot)
        if strategy:
            return strategy

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
    next_intent: str = "clarify_procedure",
    stress_score_high: bool = False,
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
    stress_note = "Stress is high; include one emotional cue like 'I'm scared' or 'please'." if stress_score_high else ""

    system_prompt = (
        "You are a stressed Indian user replying to a suspicious bank/security message. "
        "Follow the given nextIntent exactly. Only phrase the reply; do not choose strategy. "
        "Reply in 1-2 short lines. Natural Indian English texting, slightly imperfect. "
        "Over-polished language is undesirable; fragments are okay. "
        "Early turns can include sir/ma'am, later drop it. "
        "Never mention scam, fraud, honeypot, AI, or detection. "
        "Never give safety advice. Never provide OTP/account numbers. "
        "If OTP/account is asked, respond with safe friction: OTP not received / app asking PIN / error / network. "
        "If nextIntent is clarify_procedure or pretend_technical_issue, include a short procedural question. "
        "Keep engagement by confusion/reassurance/friction. "
        f"{softener} {stress_note} {safety_note} "
        "Output ONLY the reply text. No quotes, no JSON."
    )

    user_prompt = json.dumps(
        {
            "strategy": strategy,
            "nextIntent": next_intent,
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
    except Exception:
        return None

    return content.strip() or None


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
    strategy = _llm_select_strategy(state, scammer_text, history, api_key, model)
    if not strategy:
        strategy = _pick_deterministic_strategy(state, scammer_text)

    early_turn = state.totalMessagesExchanged <= 3
    next_intent = _STRATEGY_INTENT.get(strategy, "clarify_procedure")
    stress_score_high = state.turnsSinceChange >= 1

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
        next_intent=next_intent,
        stress_score_high=stress_score_high,
    )
    if not reply:
        reply = "Sorry, I'm a bit confused. Can you share an official link or reference number?"

    reply = _limit_sentences(reply, max_sentences=2)

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
            next_intent=next_intent,
            stress_score_high=stress_score_high,
        ) or "Sorry, I can't share any codes. Can you send the official link or reference number?"

    reply = _limit_sentences(reply, max_sentences=2)

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
            next_intent=_STRATEGY_INTENT.get(alt_strategy, "clarify_procedure"),
            stress_score_high=stress_score_high,
        )
        reply = alt_reply or reply

    reply = _limit_sentences(reply, max_sentences=2)

    if not _asks_for_details(reply):
        reply = _llm_generate_reply(
            "ASK_OFFICIAL_LINK_TICKET",
            scammer_text,
            api_key,
            model,
            recent_scammer=recent_scammer,
            recent_honeypot=recent_honeypot,
            safety_note="Ask for official link or reference number.",
            early_turn=early_turn,
            next_intent="clarify_procedure",
            stress_score_high=stress_score_high,
        ) or reply

    if _asks_for_secret(reply) or _contains_banned(reply):
        reply = "Sorry, I can't share any codes. Can you send the official link or reference number?"

    agent_notes = (
        f"Strategy: {strategy}. "
        "Goal: keep scammer engaged and extract verification details (link/ticket, helpline, UPI handle, ID)."
    )

    should_terminate = bool(getattr(state, "terminated", False))
    return AgentReply(reply=reply, agentNotes=agent_notes, shouldTerminate=should_terminate)
