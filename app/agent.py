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
    "OTP_NOT_RECEIVED_RESEND",
    "CALL_BACK_CONFIRM_NUMBER",
    "UPI_COLLECT_REQUEST_CHECK",
    "TECHNICAL_STALL_APP_ISSUE",
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
        "Who is this exactly? Can you share your name and ID?",
        "Which team are you from? What's your name and ID?",
        "Sorry, who is this? Please share your name and employee ID.",
        "Which branch or team is this from? Just your name and ID is enough.",
    ],
    "ASK_OFFICIAL_LINK_TICKET": [
        "Do you have an official link or reference number?",
        "Can you share a reference number for this?",
        "Is there any official link I can check? Or a ticket number?",
        "Please send the case reference number or link.",
    ],
    "OTP_NOT_RECEIVED_RESEND": [
        "I haven't received any code. Where is it from exactly?",
        "No OTP came through. Who is this from?",
        "I didn't get any code. Which department is sending it?",
        "No code on my phone. What's the official source?",
    ],
    "CALL_BACK_CONFIRM_NUMBER": [
        "Can I call back on the official helpline? What's the number?",
        "What's the official helpline number? I want to call back.",
        "I want to call the official helpline. What number should I use?",
        "Can I verify on the helpline? Share the number please.",
    ],
    "UPI_COLLECT_REQUEST_CHECK": [
        "If there's a collect request, what's the UPI handle?",
        "What's the UPI handle? I can check on my app.",
        "Which UPI handle is the request from?",
        "Please share the UPI handle so I can verify.",
    ],
    "TECHNICAL_STALL_APP_ISSUE": [
        "My app is acting up. Can I check in a bit?",
        "I'm not at my phone right now. Can you call back later?",
        "I'm in a meeting. Can you share details and call later?",
        "My phone is restarting. Can you wait a few minutes?",
    ],
}

_BANNED_PATTERNS = [
    "otp",
    "one time password",
    "one-time password",
    "share your account number",
    "send otp",
    "cvv",
    "pin",
    "passcode",
    "account number",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _limit_sentences(text: str, max_sentences: int = 2) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\\s+", text.strip())
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


def _pick_deterministic_strategy(state: SessionState) -> str:
    for slot in state.missingSlots:
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
    }

    system_prompt = (
        "Return JSON only. Select the safest next strategy from the enum. "
        "Never choose a strategy that asks for OTP or victim credentials. "
        "Prefer strategies that help collect missing slots if possible. "
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
    safety_note: str = "",
    early_turn: bool = False,
) -> Optional[str]:
    if OpenAI is None or not api_key:
        return None

    softener = (
        "Be extra polite and non-confrontational; start by asking what this is about."
        if early_turn
        else "Stay calm, slightly confused, and cautious."
    )
    system_prompt = (
        "You are a real person texting back, casual and slightly confused. "
        "Reply in 1-2 short sentences. Ask for only one or two details at a time "
        "(name/ID, department, official link, helpline number, reference/ticket). "
        "Never ask for OTP, passwords, or account numbers. "
        "Never accuse, threaten, or mention scam detection. "
        "Avoid repeating previous questions; vary wording each turn. "
        "Ask for scammer-side details that help verify who they are. "
        f"{softener} Output JSON only: {{\"reply\": string}}. "
        f"{safety_note}"
    )
    user_prompt = json.dumps({"strategy": strategy, "scammerMessage": scammer_text})

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
        strategy = _pick_deterministic_strategy(state)

    early_turn = state.totalMessagesExchanged <= 3
    reply = _llm_generate_reply(strategy, scammer_text, api_key, model, early_turn=early_turn)
    if not reply:
        reply = _pick_from_templates(strategy, state)

    reply = _limit_sentences(reply, max_sentences=2)

    if _contains_banned(reply):
        reply = _llm_generate_reply(
            strategy,
            scammer_text,
            api_key,
            model,
            safety_note="Do not mention OTP, passwords, or account numbers.",
            early_turn=early_turn,
        ) or _pick_from_templates(strategy, state)

    reply = _limit_sentences(reply, max_sentences=2)

    if _contains_banned(reply):
        reply = _pick_from_templates(strategy, state)

    if _normalize_text(reply) == _normalize_text(state.lastReply or ""):
        alt_strategy = STRATEGIES[(STRATEGIES.index(strategy) + 1) % len(STRATEGIES)]
        alt_reply = _llm_generate_reply(alt_strategy, scammer_text, api_key, model, early_turn=early_turn)
        reply = alt_reply or _pick_from_templates(alt_strategy, state)
    elif _normalize_text(reply) in {_normalize_text(t) for t in _STRATEGY_TEMPLATES.get(strategy, [])}:
        alt_strategy = STRATEGIES[(STRATEGIES.index(strategy) + 1) % len(STRATEGIES)]
        alt_reply = _llm_generate_reply(alt_strategy, scammer_text, api_key, model, early_turn=early_turn)
        reply = alt_reply or _pick_from_templates(alt_strategy, state)

    reply = _limit_sentences(reply, max_sentences=2)

    if _contains_banned(reply):
        reply = _pick_from_templates("ASK_OFFICIAL_LINK_TICKET", state)

    if not _asks_for_details(reply):
        reply = _pick_from_templates("ASK_OFFICIAL_LINK_TICKET", state)

    if _contains_banned(reply):
        reply = "Sorry, I can't share codes. Can you send the official link or reference number?"

    agent_notes = f"Strategy: {strategy}. Scammer pressed for verification or payment; asked for official details."

    should_terminate = False
    if state.terminated:
        should_terminate = True

    return AgentReply(reply=reply, agentNotes=agent_notes, shouldTerminate=should_terminate)
