from __future__ import annotations

import hashlib
from typing import List

from .models import AgentReply, SessionState


_CONFUSED_REPLIES = [
    "Sorry, I'm a bit lost. What's this about?",
    "I don't remember signing up. Can you explain?",
    "Wait, which account are you talking about?",
]

_CLARIFY_REPLIES = [
    "I got your message but I'm not sure what you need.",
    "Why do I have to do this now?",
    "Is this about a bill or something else?",
]

_STALL_REPLIES = [
    "I'm driving. Can I check later?",
    "I need a minute to look at this.",
    "Let me see, I'm not at my computer.",
]

_PROBE_REPLIES = [
    "I got a warning, but it didn't say why.",
    "It says my account is blocked. What should I do?",
    "I'm worried this is a mistake. Can you explain?",
]


def _pick_reply(options: List[str], state: SessionState) -> str:
    if not options:
        return "Okay."

    seed = f"{state.sessionId}:{state.totalMessages}"
    index = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(options)
    reply = options[index]
    if reply == state.lastReply:
        reply = options[(index + 1) % len(options)]
    return reply


def build_agent_reply(state: SessionState, scammer_text: str) -> AgentReply:
    lowered = (scammer_text or "").lower()

    if "otp" in lowered or "password" in lowered:
        reply = "I don't share codes. Why do you need that?"
    elif "upi" in lowered or "bank" in lowered or "account" in lowered or "payment" in lowered:
        reply = "I don't see anything on my side. What is this for?"
    elif "link" in lowered or "click" in lowered or "http" in lowered:
        reply = "I can't open links right now. What's it about?"
    elif "bot" in lowered:
        reply = "No, I'm just confused. Who is this?"
    else:
        bucket = [
            _pick_reply(_PROBE_REPLIES, state),
            _pick_reply(_CLARIFY_REPLIES, state),
            _pick_reply(_CONFUSED_REPLIES, state),
            _pick_reply(_STALL_REPLIES, state),
        ]
        reply = bucket[state.totalMessages % len(bucket)]

    agent_notes = "Scammer pressed for verification or payment; user remained cautious and asked for clarification."

    should_terminate = False
    if state.terminated:
        should_terminate = True

    return AgentReply(reply=reply, agentNotes=agent_notes, shouldTerminate=should_terminate)
