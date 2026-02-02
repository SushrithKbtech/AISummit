from __future__ import annotations

import re
from typing import List, Set

from .models import ExtractionResult


_BANK_PATTERN = re.compile(r"\b(?:\d[ -]?){9,18}\b")
_UPI_PATTERN = re.compile(r"\b[a-z0-9.\-_]{2,}@[a-z]{2,}\b", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\s\-]{7,}\d)")

_SUSPICIOUS_KEYWORDS = {
    "urgent",
    "verify",
    "verification",
    "account",
    "blocked",
    "suspended",
    "kyc",
    "otp",
    "payment",
    "transfer",
    "upi",
    "gift card",
    "bitcoin",
    "crypto",
    "wire",
    "bank",
    "police",
    "court",
}


def _normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    return digits


def _normalize_account(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    return digits


def extract_intelligence(text: str) -> ExtractionResult:
    normalized = text or ""

    bank_accounts: Set[str] = set(_normalize_account(match) for match in _BANK_PATTERN.findall(normalized))
    upi_ids: Set[str] = set(match.lower() for match in _UPI_PATTERN.findall(normalized))
    phishing_links: Set[str] = set(match.lower().rstrip(").,;!") for match in _URL_PATTERN.findall(normalized))
    phone_numbers: Set[str] = set(_normalize_phone(match) for match in _PHONE_PATTERN.findall(normalized))

    # Avoid misclassifying phone numbers as bank accounts.
    bank_accounts = {
        account
        for account in bank_accounts
        if account not in phone_numbers and len(account) != 10
    }

    lowered = normalized.lower()
    suspicious_keywords: Set[str] = set()
    for keyword in _SUSPICIOUS_KEYWORDS:
        if keyword in lowered:
            suspicious_keywords.add(keyword)

    return ExtractionResult(
        bankAccounts=sorted(bank_accounts),
        upiIds=sorted(upi_ids),
        phishingLinks=sorted(phishing_links),
        phoneNumbers=sorted(phone_numbers),
        suspiciousKeywords=sorted(suspicious_keywords),
    )


def merge_extraction(existing: ExtractionResult, incoming: ExtractionResult) -> ExtractionResult:
    return ExtractionResult(
        bankAccounts=sorted(set(existing.bankAccounts).union(incoming.bankAccounts)),
        upiIds=sorted(set(existing.upiIds).union(incoming.upiIds)),
        phishingLinks=sorted(set(existing.phishingLinks).union(incoming.phishingLinks)),
        phoneNumbers=sorted(set(existing.phoneNumbers).union(incoming.phoneNumbers)),
        suspiciousKeywords=sorted(set(existing.suspiciousKeywords).union(incoming.suspiciousKeywords)),
    )
