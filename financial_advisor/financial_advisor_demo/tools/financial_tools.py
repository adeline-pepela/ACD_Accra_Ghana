"""
tools/financial_tools.py
-------------------------
This is the real, unmodified Strands Agents SDK `@tool` decorator (from the
`strands-agents` PyPI package -- pip install strands-agents) wrapping our mock
data functions. Nothing here is reimplemented: `@tool` parses each function's
type hints into a JSON schema and turns the docstring into the tool
description, exactly as it does in the AWS reference architecture. Swap the
bodies for real API calls (a brokerage API, a market-data vendor, a news
vendor) and this file is production code, unchanged.
"""

from typing import Optional

from strands import tool

from financial_advisor_demo.mock_data import (
    fetch_portfolio_data,
    fetch_market_prices,
    fetch_news_feed,
    compute_risk_metrics,
    generate_recommendations,
)


@tool
def get_portfolio_value(client_id: str, include_unrealized_gains: bool = True) -> dict:
    """Retrieve the current portfolio value and positions for a client."""
    return fetch_portfolio_data(client_id, include_unrealized_gains)


@tool
def get_market_data(tickers: list[str], period: str = "1d") -> dict:
    """Retrieve current market prices and key metrics for a list of tickers."""
    return fetch_market_prices(tickers, period)


@tool
def get_financial_news(query: str, max_results: int = 5) -> dict:
    """Retrieve recent financial news articles and sentiment relevant to a query."""
    return fetch_news_feed(query, max_results)


@tool
def get_risk_analysis(client_id: str, risk_tolerance: Optional[str] = None) -> dict:
    """Perform a risk analysis on a client's portfolio."""
    return compute_risk_metrics(client_id, risk_tolerance)


@tool
def get_investment_recommendations(
    client_id: str, goal: str = "growth", time_horizon_years: int = 10
) -> dict:
    """Generate personalised investment recommendations for a client."""
    return generate_recommendations(client_id, goal, time_horizon_years)


ALL_TOOLS = [
    get_portfolio_value,
    get_market_data,
    get_financial_news,
    get_risk_analysis,
    get_investment_recommendations,
]
