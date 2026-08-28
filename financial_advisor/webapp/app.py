"""
webapp/app.py
--------------
A small Flask web application that puts a browser-based chat UI in front of
the same `financial_advisor_demo` package used by the Jupyter notebook demo.

No new agent logic lives here -- this file only adds a web layer (routes +
per-browser-session state) around `create_financial_advisor()`. The
orchestrator, specialists, memory manager, and guardrails are imported
unchanged from the sibling `financial_advisor_demo/` package.
"""

from __future__ import annotations

import os
import sys
import uuid

from flask import Flask, jsonify, render_template, request, session

# financial_advisor_demo/ lives one directory up from this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_advisor_demo.agents.orchestrator import create_financial_advisor
from financial_advisor_demo.guardrails.config import GuardrailsEngine
from financial_advisor_demo.memory.memory_manager import MemoryManager
from financial_advisor_demo.mock_data import _CLIENTS

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-secret-not-for-production")

# In-memory per-browser-session store: {session_id: FinancialAdvisor}.
# Fine for a single-process demo; a production deploy would use AgentCore
# Runtime's own session handling instead.
_SESSIONS: dict[str, object] = {}


def _get_advisor():
    sid = session.get("sid")
    if not sid or sid not in _SESSIONS:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _SESSIONS[sid] = create_financial_advisor(
            memory=MemoryManager(use_local_fallback=True),
            guardrails=GuardrailsEngine(),
        )
    return _SESSIONS[sid]


@app.route("/")
def index():
    clients = [
        {"id": cid, "name": c["name"], "risk": c["risk_tolerance"]}
        for cid, c in _CLIENTS.items()
    ]
    return render_template("index.html", clients=clients)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    client_id = (data.get("client_id") or "ABC123").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    advisor = _get_advisor()
    # The demo's client_id extraction reads it straight out of the query text,
    # so we prepend it here based on the dropdown selection -- the user
    # doesn't have to type "I'm client ABC123" themselves every time.
    augmented_query = f"I'm client {client_id}. {message}"
    result = advisor.ask(augmented_query)
    result["client_id"] = client_id
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid and sid in _SESSIONS:
        del _SESSIONS[sid]
    session.pop("sid", None)
    return jsonify({"ok": True})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "active_sessions": len(_SESSIONS)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"Financial Advisor demo running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
