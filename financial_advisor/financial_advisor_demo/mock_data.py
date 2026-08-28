"""
mock_data.py
------------
Deterministic, realistic dummy data providers that stand in for real brokerage
/ market-data / news APIs. In the AWS reference build (awslabs/agentcore-samples,
finance-personal-assistant) these same function names sit behind boto3 calls to
real services; here they return seeded synthetic data so the whole demo runs
with zero AWS credentials and zero network calls,
`python main.py` local-mock mode.

Every function is deterministic per client_id / ticker (seeded RNG), so the
same client always has the same "current" portfolio within a run, and the
demo is reproducible for recordings.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Optional

_CLIENTS = {
    "ABC123": {
        "name": "Thandiwe Nkosi",
        "risk_tolerance": "moderate",
        "holdings": {"AAPL": 120, "MSFT": 80, "GOOGL": 40, "AMZN": 25, "SPY": 200},
        "cash_balance": 45_000.00,
    },
    "XYZ789": {
        "name": "Sipho Dlamini",
        "risk_tolerance": "aggressive",
        "holdings": {"TSLA": 60, "NVDA": 90, "AMZN": 30, "META": 50},
        "cash_balance": 12_500.00,
    },
}

_PRICE_BOOK = {
    "AAPL": 231.40, "MSFT": 421.10, "GOOGL": 172.85, "AMZN": 198.60,
    "SPY": 566.20, "TSLA": 248.30, "NVDA": 138.90, "META": 592.10,
}

_52W_RANGE = {
    "AAPL": (164.08, 260.10), "MSFT": (309.45, 468.35), "GOOGL": (130.67, 207.05),
    "AMZN": (151.61, 242.52), "SPY": (440.13, 610.78), "TSLA": (138.80, 488.54),
    "NVDA": (75.61, 153.13), "META": (414.50, 638.40),
}


def _client_rng(client_id: str) -> random.Random:
    return random.Random(f"seed-{client_id}")


def fetch_portfolio_data(client_id: str, include_unrealized_gains: bool = True) -> dict:
    """Simulated brokerage lookup: current positions, market value, P&L."""
    client = _CLIENTS.get(client_id)
    if not client:
        return {"error": f"No client record found for '{client_id}'"}

    rng = _client_rng(client_id)
    positions = []
    total_value = client["cash_balance"]
    for ticker, shares in client["holdings"].items():
        price = _PRICE_BOOK[ticker] * (1 + rng.uniform(-0.01, 0.01))
        market_value = round(price * shares, 2)
        total_value += market_value
        entry = {
            "ticker": ticker,
            "shares": shares,
            "price": round(price, 2),
            "market_value": market_value,
        }
        if include_unrealized_gains:
            cost_basis = round(price * shares * rng.uniform(0.75, 0.98), 2)
            entry["cost_basis"] = cost_basis
            entry["unrealized_gain"] = round(market_value - cost_basis, 2)
            entry["unrealized_gain_pct"] = round((market_value - cost_basis) / cost_basis * 100, 2)
        positions.append(entry)

    return {
        "client_id": client_id,
        "client_name": client["name"],
        "cash_balance": client["cash_balance"],
        "positions": positions,
        "total_portfolio_value": round(total_value, 2),
        "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }


def compute_risk_metrics(client_id: str, risk_tolerance: Optional[str] = None) -> dict:
    """Simulated risk engine: concentration, volatility, beta, diversification score."""
    client = _CLIENTS.get(client_id)
    if not client:
        return {"error": f"No client record found for '{client_id}'"}

    rng = _client_rng(client_id)
    portfolio = fetch_portfolio_data(client_id)
    tolerance = risk_tolerance or client["risk_tolerance"]

    weights = {
        p["ticker"]: p["market_value"] / portfolio["total_portfolio_value"]
        for p in portfolio["positions"]
    }
    top_holding, top_weight = max(weights.items(), key=lambda kv: kv[1])
    concentration_flag = top_weight > 0.30

    beta = round(sum(weights[t] * rng.uniform(0.8, 1.6) for t in weights), 2)
    volatility_30d = round(rng.uniform(12.0, 34.0), 1)

    return {
        "client_id": client_id,
        "risk_tolerance": tolerance,
        "portfolio_beta": beta,
        "volatility_30d_pct": volatility_30d,
        "largest_position": {"ticker": top_holding, "weight_pct": round(top_weight * 100, 1)},
        "concentration_risk": "HIGH" if concentration_flag else "NORMAL",
        "diversification_score": round(rng.uniform(4.0, 9.0), 1),
    }


def fetch_market_prices(tickers: list[str], period: str = "1d") -> dict:
    """Simulated market-data feed: live price, day change, 52-week range."""
    rng = random.Random(f"seed-market-{','.join(sorted(tickers))}-{period}")
    quotes = []
    for t in tickers:
        base = _PRICE_BOOK.get(t)
        if base is None:
            quotes.append({"ticker": t, "error": "unknown ticker"})
            continue
        change_pct = round(rng.uniform(-2.4, 2.4), 2)
        low, high = _52W_RANGE[t]
        quotes.append({
            "ticker": t,
            "price": round(base, 2),
            "change_pct": change_pct,
            "day_range": [round(base * 0.99, 2), round(base * 1.01, 2)],
            "52w_range": [low, high],
            "period": period,
        })
    return {"quotes": quotes, "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}


_HEADLINE_BANK = {
    "AAPL": [
        ("Apple supply chain signals steady iPhone demand into next quarter", 0.42),
        ("Analysts split on Apple services growth after latest guidance", 0.05),
        ("Apple faces fresh antitrust scrutiny in two major markets", -0.35),
    ],
    "MSFT": [
        ("Microsoft cloud unit posts another quarter of accelerating growth", 0.55),
        ("Microsoft AI infrastructure spend draws investor questions on margins", -0.12),
    ],
    "TSLA": [
        ("Tesla delivery numbers beat estimates for the quarter", 0.48),
        ("Tesla faces new regulatory inquiry over driver-assist claims", -0.40),
    ],
    "default": [
        ("Markets steady as investors await central bank commentary", 0.03),
        ("Volatility ticks up on mixed macro data", -0.10),
    ],
}


def fetch_news_feed(query: str, max_results: int = 5) -> dict:
    """Simulated financial news + sentiment feed."""
    rng = random.Random(f"seed-news-{query.lower()}")
    matched_key = next((k for k in _HEADLINE_BANK if k.lower() in query.lower()), "default")
    pool = _HEADLINE_BANK[matched_key] + _HEADLINE_BANK["default"]
    rng.shuffle(pool)
    articles = []
    for i, (headline, sentiment) in enumerate(pool[:max_results]):
        articles.append({
            "headline": headline,
            "sentiment": sentiment,
            "sentiment_label": "positive" if sentiment > 0.15 else "negative" if sentiment < -0.15 else "neutral",
            "published": (datetime.utcnow() - timedelta(hours=i * 3 + 1)).strftime("%Y-%m-%d %H:%M UTC"),
        })
    avg_sentiment = round(sum(a["sentiment"] for a in articles) / len(articles), 2) if articles else 0.0
    return {"query": query, "articles": articles, "average_sentiment": avg_sentiment}


def generate_recommendations(client_id: str, goal: str = "growth", time_horizon_years: int = 10) -> dict:
    """Simulated recommendation engine combining risk profile + goal + horizon."""
    client = _CLIENTS.get(client_id)
    if not client:
        return {"error": f"No client record found for '{client_id}'"}

    risk = compute_risk_metrics(client_id)
    actions = []
    if risk["concentration_risk"] == "HIGH":
        actions.append(
            f"Trim {risk['largest_position']['ticker']} \u2014 currently "
            f"{risk['largest_position']['weight_pct']}% of the portfolio, above the 30% comfort threshold."
        )
    if goal == "growth" and time_horizon_years >= 7:
        actions.append("Maintain equity-heavy allocation; horizon supports riding out short-term volatility.")
    elif goal == "income":
        actions.append("Shift a portion of equity gains into dividend-paying or fixed-income positions.")
    if client["risk_tolerance"] == "conservative" and risk["volatility_30d_pct"] > 20:
        actions.append("Current volatility exceeds conservative comfort band \u2014 consider partial rebalancing.")
    if not actions:
        actions.append("Portfolio is broadly aligned with stated goal and risk tolerance \u2014 no urgent changes.")

    return {
        "client_id": client_id,
        "goal": goal,
        "time_horizon_years": time_horizon_years,
        "recommended_actions": actions,
    }
