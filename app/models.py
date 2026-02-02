from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

Sender = Literal["scammer", "user"]


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sender: Sender
    text: str = Field(default="")
    timestamp: str


class Metadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    channel: Optional[str] = None
    language: Optional[str] = None
    locale: Optional[str] = None


class IncomingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sessionId: str
    message: Message
    conversationHistory: Optional[List[Message]] = None
    metadata: Optional[Metadata] = None


class ReplyResponse(BaseModel):
    status: str
    reply: str


class ErrorResponse(BaseModel):
    # GUVI expects "message" (NOT "error")
    status: str
    message: str


class Intelligence(BaseModel):
    bankAccounts: List[str]
    upiIds: List[str]
    phishingLinks: List[str]
    phoneNumbers: List[str]
    suspiciousKeywords: List[str]


class FinalCallbackPayload(BaseModel):
    sessionId: str
    scamDetected: bool
    totalMessagesExchanged: int
    extractedIntelligence: Intelligence
    agentNotes: str


class SessionState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sessionId: str
    scamScore: float = 0.0
    scamConfirmed: bool = False
    agentActive: bool = False
    totalMessagesExchanged: int = 0
    lastReply: Optional[str] = None
    turnsSinceChange: int = 0
    terminated: bool = False
    finalCallbackSent: bool = False
    extractedIntelligence: Intelligence
    lastScammerMessage: Optional[str] = None
    agentNotes: str = ""
    missingSlots: List[str] = []

    # Rolling memory (works even when GUVI doesn’t send conversationHistory)
    recentScammer: List[str] = []
    recentHoneypot: List[str] = []


class ExtractionResult(BaseModel):
    bankAccounts: List[str]
    upiIds: List[str]
    phishingLinks: List[str]
    phoneNumbers: List[str]
    suspiciousKeywords: List[str]


class DetectorResult(BaseModel):
    score: float
    indicators: List[str]


class AgentReply(BaseModel):
    reply: str
    agentNotes: str
    shouldTerminate: bool
