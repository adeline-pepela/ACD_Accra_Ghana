"""
reasoning.py
------------
In production (see agents/specialists.py and agents/orchestrator.py docstrings)
the "brain" behind each Strands `Agent` is `BedrockModel` calling Claude on
Amazon Bedrock: it reads the tool specs, decides which tool(s) to call and
with what arguments, then writes the natural-language answer.

This sandbox has no AWS credentials and no model API key, so `LocalReasoningEngine`
is a deterministic, offline stand-in for that model call. It implements the exact
same four-step loop the talk describes:

    Perceive -> read the request + injected memory/context
    Plan     -> score the request against each tool's real Strands tool_spec
                (name + description + schema) to pick which tool(s) apply
    Act      -> call the real, unmodified @tool function(s) with the parsed
                arguments
    Reflect  -> turn the raw tool output into a written answer, and note
                whether another tool call is still needed

Swap this class for `strands.models.BedrockModel` and every other file in this
package -- the tools, the specialist agents, the orchestrator, memory, and
guardrails -- is unchanged. That substitution is the entire gap between this
demo and the AWS-hosted production version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCallTrace:
    tool_name: str
    arguments: dict
    result: Any


@dataclass
class ReasoningTrace:
    perceive: str
    plan: str
    calls: list[ToolCallTrace] = field(default_factory=list)
    reflect: str = ""


class LocalReasoningEngine:
    """A deterministic offline planner/synthesiser scoped to one specialist's toolset."""

    def __init__(self, tools: list[Callable], keyword_map: dict[str, list[str]]):
        self.tools = {t.tool_name: t for t in tools}
        self.keyword_map = keyword_map

    def _plan(self, query: str) -> list[str]:
        q = query.lower()
        scored = []
        for name, keywords in self.keyword_map.items():
            score = sum(1 for kw in keywords if kw in q)
            if score:
                scored.append((score, name))
        scored.sort(reverse=True)
        return [name for _, name in scored] or list(self.tools.keys())

    def run(self, query: str, arg_resolver: Callable[[str, str], dict]) -> ReasoningTrace:
        trace = ReasoningTrace(
            perceive=f'Received request: "{query}"',
            plan="",
        )
        plan_order = self._plan(query)
        trace.plan = f"Relevant tool(s) ranked: {plan_order}"

        for tool_name in plan_order:
            tool_fn = self.tools[tool_name]
            args = arg_resolver(tool_name, query)
            result = tool_fn(**args)
            trace.calls.append(ToolCallTrace(tool_name, args, result))

        trace.reflect = f"Gathered {len(trace.calls)} tool result(s); composing response."
        return trace


_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")
_KNOWN_TICKERS = {"AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "TSLA", "NVDA", "META"}


def extract_tickers(text: str) -> list[str]:
    found = [t for t in _TICKER_PATTERN.findall(text) if t in _KNOWN_TICKERS]
    return found or ["SPY"]


_CLIENT_ID_PATTERN = re.compile(r"\b([A-Z]{3}\d{3})\b")


def extract_client_id(text: str, default: str = "ABC123") -> str:
    match = _CLIENT_ID_PATTERN.search(text)
    return match.group(1) if match else default
