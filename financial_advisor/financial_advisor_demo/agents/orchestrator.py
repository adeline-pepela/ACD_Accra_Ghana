"""
agents/orchestrator.py
------------------------
Production version wraps each specialist as a `@tool` function and hands them
to a top-level `strands.Agent` (the "agent-as-tool" pattern) -- see the AWS
reference build:

    @tool
    def ask_portfolio_agent(query: str) -> str:
        # docstring: "Delegate a portfolio analysis or risk question to the
        # Portfolio Specialist Agent."
        return str(portfolio_agent(query))

    def create_financial_advisor(guardrail_id=None, memory_context="", ...):
        return Agent(model=BedrockModel(...), system_prompt=ORCHESTRATOR_PROMPT,
                     tools=[ask_portfolio_agent, ask_market_data_agent,
                            ask_news_agent, get_investment_recommendations])

There, Claude (via BedrockModel) reads the delegate tools' docstrings and
decides which specialist(s) a question needs. Here, `FinancialAdvisor` uses
the same real `@tool`-wrapped delegate functions (so their schemas are
genuine Strands tool specs -- check `.tool_spec` on any of them) but routes
to them with the same keyword-scoring approach as the specialists, via
`LocalReasoningEngine`. The orchestration *shape* -- delegate, gather,
synthesise, run through guardrails, persist to memory -- is identical to
production.
"""

from __future__ import annotations

from strands import tool

from financial_advisor_demo.agents.specialists import (
    create_portfolio_agent,
    create_market_data_agent,
    create_news_agent,
)
from financial_advisor_demo.tools.financial_tools import get_investment_recommendations
from financial_advisor_demo.reasoning import extract_client_id
from financial_advisor_demo.guardrails.config import GuardrailsEngine
from financial_advisor_demo.memory.memory_manager import MemoryManager

ORCHESTRATOR_PROMPT = """
You are a Senior Financial Advisor AI. Your role is to provide comprehensive,
personalised financial guidance. You have three specialist agents available:

- Portfolio Specialist: portfolio values, positions, risk metrics
- Market Data Specialist: prices, indices, volatility
- News Analyst: financial news and sentiment

Always delegate to the right specialist(s), then synthesise their findings
into a clear, actionable response for the client.
"""

_ROUTING_KEYWORDS = {
    "portfolio": ["portfolio", "holdings", "position", "worth", "value", "risk", "concentrat", "beta", "diversif"],
    "market": ["price", "market", "trading", "quote", "index", "indices"],
    "news": ["news", "headline", "sentiment", "article"],
    "recommend": ["rebalance", "recommend", "should i", "advice", "what should"],
}


class FinancialAdvisor:
    """The orchestrator. Delegates to specialists-as-tools, then synthesises."""

    def __init__(self, memory: MemoryManager, guardrails: GuardrailsEngine):
        self.memory = memory
        self.guardrails = guardrails
        self._portfolio_agent = create_portfolio_agent()
        self._market_agent = create_market_data_agent()
        self._news_agent = create_news_agent()
        self.delegate_tools = self._build_delegate_tools()
        self.last_delegation_log: list = []

    def _build_delegate_tools(self):
        portfolio_agent, market_agent, news_agent = (
            self._portfolio_agent, self._market_agent, self._news_agent
        )

        @tool
        def ask_portfolio_agent(query: str) -> str:
            """Delegate a portfolio analysis or risk question to the Portfolio Specialist Agent."""
            return portfolio_agent.ask(query)

        @tool
        def ask_market_data_agent(query: str) -> str:
            """Delegate a market data or pricing question to the Market Data Specialist Agent."""
            return market_agent.ask(query)

        @tool
        def ask_news_agent(query: str) -> str:
            """Delegate a news or sentiment analysis question to the News Analyst Agent."""
            return news_agent.ask(query)

        return {
            "portfolio": ask_portfolio_agent,
            "market": ask_market_data_agent,
            "news": ask_news_agent,
        }

    def _route(self, query: str):
        lowered = query.lower()
        recommend_kws = _ROUTING_KEYWORDS["recommend"]
        matched = []
        for domain, kws in _ROUTING_KEYWORDS.items():
            if domain == "recommend":
                continue
            if any(kw in lowered for kw in kws):
                matched.append(domain)
        needs_recommendation = any(kw in lowered for kw in recommend_kws)
        if needs_recommendation and not matched:
            matched = ["portfolio", "market", "news"]
        return matched, needs_recommendation

    def ask(self, query: str) -> dict:
        input_verdict = self.guardrails.check_input(query)
        if not input_verdict.allowed:
            return {"blocked": True, "stage": "input", "reason": input_verdict.reason, "response": None}

        safe_query = input_verdict.redacted_text
        memory_context = self.memory.get_recent_context(query=safe_query)

        domains, needs_recommendation = self._route(safe_query)
        self.last_delegation_log = []
        specialist_snippets = []
        for domain in domains:
            tool_fn = self.delegate_tools[domain]
            result = tool_fn(query=safe_query)
            self.last_delegation_log.append(f"{tool_fn.tool_name} -> {result[:120]}...")
            specialist_snippets.append(result)

        recommendation_snippet = ""
        if needs_recommendation:
            client_id = extract_client_id(safe_query)
            rec = get_investment_recommendations(client_id=client_id)
            self.last_delegation_log.append(f"get_investment_recommendations -> {rec}")
            if "error" not in rec:
                recommendation_snippet = "Recommended actions: " + " ".join(rec["recommended_actions"])

        personalization = ""
        if memory_context:
            useful_lines = [
                ln for ln in memory_context.splitlines()
                if ln.strip() and not ln.startswith("Previous conversation context:")
            ]
            if useful_lines:
                personalization = f"\n\n(Drawing on memory: {useful_lines[0]})"

        draft_response = " ".join(s for s in specialist_snippets if s)
        if recommendation_snippet:
            draft_response = f"{draft_response} {recommendation_snippet}".strip()
        draft_response = (draft_response or "I don't have enough information to answer that yet.") + personalization

        output_verdict = self.guardrails.check_output(draft_response)
        if not output_verdict.allowed:
            return {"blocked": True, "stage": "output", "reason": output_verdict.reason, "response": None}

        final_response = output_verdict.redacted_text
        self.memory.save_turn(safe_query, final_response)

        return {
            "blocked": False,
            "domains_consulted": domains,
            "delegation_log": self.last_delegation_log,
            "response": final_response,
        }


def create_financial_advisor(memory: MemoryManager = None,
                              guardrails: GuardrailsEngine = None) -> FinancialAdvisor:
    return FinancialAdvisor(
        memory=memory or MemoryManager(use_local_fallback=True),
        guardrails=guardrails or GuardrailsEngine(),
    )
