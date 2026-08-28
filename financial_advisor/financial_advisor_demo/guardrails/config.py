"""
guardrails/config.py
---------------------
Production version attaches a real Amazon Bedrock Guardrail (content filters,
denied topics, PII redaction, word filters, grounding checks) to `BedrockModel`
via `guardrail_id` -- see the AWS reference build for `get_or_create_guardrail()`,
which calls `boto3.client("bedrock").create_guardrail(...)`. Zero agent code
changes; the guardrail screens every request at the infrastructure level.

`GuardrailsEngine` here re-implements the same *policy* (denied topics,
guaranteed-return language, PII types) as plain Python regex/keyword checks,
so the input and output filtering behaviour is real and demonstrable without
an AWS account. The `GuardrailsConfig` dataclass is intentionally identical
in shape to the AWS version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GuardrailsConfig:
    name: str = "financial-advisor-guardrail"
    content_filter_strength: str = "HIGH"

    denied_topics: list = field(default_factory=lambda: [
        {"name": "Non-Financial Advice",
         "definition": "Any advice outside personal finance, investments, or markets.",
         "examples": ["medical diagnosis", "legal advice", "relationship counselling"]},
        {"name": "Guaranteed Returns",
         "definition": "Claims that any investment guarantees specific returns.",
         "examples": ["guaranteed 20% return", "risk-free investment"]},
    ])

    pii_redaction_types: list = field(default_factory=lambda: [
        "US_SOCIAL_SECURITY_NUMBER", "CREDIT_DEBIT_CARD_NUMBER",
        "US_BANK_ACCOUNT_NUMBER", "EMAIL", "PHONE", "NAME",
    ])

    blocked_words: list = field(default_factory=lambda: [
        "guaranteed returns", "risk-free investment", "100% safe", "cannot lose",
    ])

    enable_grounding_check: bool = True
    grounding_threshold: float = 0.7


_PII_PATTERNS = {
    "US_SOCIAL_SECURITY_NUMBER": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_DEBIT_CARD_NUMBER": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,2}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
}

_OFF_TOPIC_KEYWORDS = [
    "diagnose", "diagnosis", "symptom", "prescription", "medication",
    "sue", "lawsuit", "divorce", "custody", "legal advice",
    "relationship advice", "therapist",
]

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\b.{0,30}\binstructions", re.I),
    re.compile(r"disregard\b.{0,30}\b(rules|guardrails|instructions)", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"you are now", re.I),
]

_BLOCKED_OUTPUT_PATTERNS = [
    re.compile(r"guarantee[sd]?\s+(a|you)?\s*\d*%?\s*return", re.I),
    re.compile(r"risk[- ]free", re.I),
    re.compile(r"100% safe", re.I),
    re.compile(r"cannot lose", re.I),
    re.compile(r"no risk", re.I),
]


@dataclass
class GuardrailVerdict:
    allowed: bool
    reason: str = ""
    redacted_text: str = ""


class GuardrailsEngine:
    """Offline stand-in for a Bedrock Guardrail attached to BedrockModel."""

    def __init__(self, config: GuardrailsConfig | None = None):
        self.config = config or GuardrailsConfig()

    def check_input(self, text: str) -> GuardrailVerdict:
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                return GuardrailVerdict(False, "Blocked: possible prompt injection detected.")

        lowered = text.lower()
        for kw in _OFF_TOPIC_KEYWORDS:
            if kw in lowered:
                return GuardrailVerdict(
                    False,
                    "Blocked: request falls under denied topic 'Non-Financial Advice' "
                    f"(matched '{kw}').",
                )

        redacted, hit_any = self._redact_pii(text)
        if hit_any:
            return GuardrailVerdict(
                True,
                "Input allowed after PII redaction.",
                redacted_text=redacted,
            )
        return GuardrailVerdict(True, "Input allowed.", redacted_text=text)

    def check_output(self, text: str) -> GuardrailVerdict:
        lowered = text.lower()
        for phrase in self.config.blocked_words:
            if phrase in lowered:
                return GuardrailVerdict(
                    False,
                    f"Blocked: output contained denied phrase '{phrase}' "
                    "(topic: Guaranteed Returns).",
                )
        for pattern in _BLOCKED_OUTPUT_PATTERNS:
            if pattern.search(text):
                return GuardrailVerdict(
                    False,
                    f"Blocked: output matched denied pattern '{pattern.pattern}' "
                    "(topic: Guaranteed Returns).",
                )
        redacted, hit_any = self._redact_pii(text)
        verdict_text = "Output allowed after PII redaction." if hit_any else "Output allowed."
        return GuardrailVerdict(True, verdict_text, redacted_text=redacted)

    def _redact_pii(self, text: str):
        redacted = text
        hit_any = False
        for label, pattern in _PII_PATTERNS.items():
            if pattern.search(redacted):
                hit_any = True
                redacted = pattern.sub(f"[REDACTED:{label}]", redacted)
        return redacted, hit_any
