"""
agents/specialists.py
----------------------
Production version (real Strands SDK + Amazon Bedrock -- this is the actual
code from the AWS reference build; NOT executed in this offline demo):

    from strands import Agent
    from strands.models import BedrockModel

    def create_portfolio_agent(guardrail_id=None) -> Agent:
        return Agent(
            model=BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                                region_name="us-east-1", guardrail_id=guardrail_id),
            system_prompt="You are a Portfolio Specialist...",
            tools=[get_portfolio_value, get_risk_analysis],
        )

Here, each specialist is a `SpecialistAgent`: same idea -- a focused system
prompt, a narrow toolset, and a `.ask(query)` entry point that returns a
written answer -- but driven by `LocalReasoningEngine` (see reasoning.py)
instead of `BedrockModel`, so it runs with no AWS credentials.
"""

from __future__ import annotations

from financial_advisor_demo.reasoning import LocalReasoningEngine, extract_client_id, extract_tickers
from financial_advisor_demo.tools.financial_tools import (
    get_portfolio_value,
    get_risk_analysis,
    get_market_data,
    get_financial_news,
)


class SpecialistAgent:
    """Mirrors strands.Agent's shape: a system_prompt, a tools list, and a callable."""

    def __init__(self, name: str, system_prompt: str, engine: LocalReasoningEngine, synth):
        self.name = name
        self.system_prompt = system_prompt
        self.engine = engine
        self._synth = synth
        self.last_trace = None
        self._resolvers: dict = {}

    def ask(self, query: str) -> str:
        trace = self.engine.run(query, self._arg_resolver)
        self.last_trace = trace
        return self._synth(trace)

    def _arg_resolver(self, tool_name: str, query: str) -> dict:
        return self._resolvers[tool_name](query)


def create_portfolio_agent() -> SpecialistAgent:
    engine = LocalReasoningEngine(
        tools=[get_portfolio_value, get_risk_analysis],
        keyword_map={
            "get_portfolio_value": ["portfolio", "value", "holdings", "position", "worth", "balance"],
            "get_risk_analysis": ["risk", "volatil", "concentrat", "beta", "diversif"],
        },
    )

    def resolver(query):
        client_id = extract_client_id(query)
        return {"client_id": client_id}

    def synth(trace):
        lines = []
        for call in trace.calls:
            r = call.result
            if call.tool_name == "get_portfolio_value" and "error" not in r:
                lines.append(
                    f"Client {r['client_name']} ({r['client_id']}) holds a portfolio worth "
                    f"${r['total_portfolio_value']:,.2f} as of {r['as_of']}, including "
                    f"${r['cash_balance']:,.2f} in cash. Largest positions: " +
                    ", ".join(f"{p['ticker']} (${p['market_value']:,.0f})"
                              for p in sorted(r['positions'], key=lambda p: -p['market_value'])[:3]) + "."
                )
            elif call.tool_name == "get_risk_analysis" and "error" not in r:
                lines.append(
                    f"Risk profile: beta {r['portfolio_beta']}, 30-day volatility "
                    f"{r['volatility_30d_pct']}%, largest position {r['largest_position']['ticker']} at "
                    f"{r['largest_position']['weight_pct']}% of the book -- concentration risk is "
                    f"{r['concentration_risk']}. Diversification score {r['diversification_score']}/10."
                )
        return " ".join(lines) if lines else "No portfolio data available for that client."

    agent = SpecialistAgent(
        name="Portfolio Specialist",
        system_prompt=(
            "You are a Portfolio Specialist. Analyse holdings, risk metrics, and P&L. "
            "Be precise and cite specific numbers."
        ),
        engine=engine,
        synth=synth,
    )
    agent._resolvers = {"get_portfolio_value": resolver, "get_risk_analysis": resolver}
    return agent


def create_market_data_agent() -> SpecialistAgent:
    engine = LocalReasoningEngine(
        tools=[get_market_data],
        keyword_map={"get_market_data": ["price", "market", "trading", "quote", "index", "indices"]},
    )

    def resolver(query):
        return {"tickers": extract_tickers(query)}

    def synth(trace):
        lines = []
        for call in trace.calls:
            for q in call.result.get("quotes", []):
                if "error" in q:
                    continue
                direction = "up" if q["change_pct"] >= 0 else "down"
                lines.append(
                    f"{q['ticker']}: ${q['price']:,.2f}, {direction} {abs(q['change_pct'])}% today "
                    f"(52-week range ${q['52w_range'][0]:,.2f}-${q['52w_range'][1]:,.2f})."
                )
        return " ".join(lines) if lines else "No market data available for that request."

    agent = SpecialistAgent(
        name="Market Data Specialist",
        system_prompt=(
            "You are a Market Data Specialist. Provide current prices, 52-week ranges, "
            "and key indices. Stick to facts."
        ),
        engine=engine,
        synth=synth,
    )
    agent._resolvers = {"get_market_data": resolver}
    return agent


def create_news_agent() -> SpecialistAgent:
    engine = LocalReasoningEngine(
        tools=[get_financial_news],
        keyword_map={"get_financial_news": ["news", "headline", "sentiment", "article", "latest"]},
    )

    def resolver(query):
        return {"query": query, "max_results": 3}

    def synth(trace):
        lines = []
        for call in trace.calls:
            r = call.result
            if not r.get("articles"):
                continue
            label = "positive" if r["average_sentiment"] > 0.15 else "negative" if r["average_sentiment"] < -0.15 else "mixed"
            headlines = "; ".join(f'"{a["headline"]}" ({a["sentiment_label"]})' for a in r["articles"])
            lines.append(f"Recent headlines -- {headlines}. Overall sentiment: {label} ({r['average_sentiment']:+.2f}).")
        return " ".join(lines) if lines else "No relevant news found."

    agent = SpecialistAgent(
        name="News Analyst",
        system_prompt=(
            "You are a Financial News Analyst. Summarise recent news and provide "
            "sentiment analysis. Be objective."
        ),
        engine=engine,
        synth=synth,
    )
    agent._resolvers = {"get_financial_news": resolver}
    return agent
