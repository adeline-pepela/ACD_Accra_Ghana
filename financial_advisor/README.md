# AWS Community Day Accra, Ghana 🇬🇭

## Building a Memory-Powered Multi-Agent Financial Advisor with Strands SDK & Amazon Bedrock

This repository contains the demo code and resources from the talk presented at **AWS Community Day Accra, Ghana**.

---

## Talk Overview

This session walks through building a production-ready multi-agent financial advisor using:

- **Strands Agents SDK** — open-source Python framework for building agents
- **Amazon Bedrock** — foundation models (Anthropic Claude)
- **Amazon Bedrock AgentCore** — memory, guardrails, and runtime deployment
- **Amazon Bedrock Guardrails** — responsible AI and compliance controls

**Content Level:** 300 (Advanced)

---

## What We Built

A memory-powered multi-agent financial advisor that delegates to specialist sub-agents:

| Specialist | Responsibility |
|---|---|
| Portfolio Specialist | Holdings, risk metrics, P&L |
| Market Data Specialist | Live prices, 52-week ranges |
| News Analyst | Headlines, sentiment analysis |
| Senior Financial Advisor (Orchestrator) | Synthesizes all specialists into one coherent answer |

---

## Key Concepts Covered

- **Agent-as-tool pattern** — sub-agents wrapped as tools for the orchestrator
- **Persistent memory** — Amazon Bedrock AgentCore Memory for cross-session context
- **Guardrails as infrastructure** — prompt injection protection, PII redaction, compliance controls
- **Local demo mode** — fully runnable offline without AWS credentials
- **Production deployment** — `agentcore deploy` for serverless, auto-scaling endpoints

---

## Project Structure

```
financial_advisor/
├── Multi_Agent_Financial_Advisor_Demo.ipynb   # Main demo notebook
├── financial_advisor_demo/
│   ├── agents/
│   │   ├── orchestrator.py                    # Senior Financial Advisor
│   │   └── specialists.py                     # Portfolio, Market, News agents
│   ├── guardrails/
│   │   └── config.py                          # Guardrails engine
│   ├── memory/
│   │   └── memory_manager.py                  # AgentCore Memory wrapper
│   ├── tools/
│   │   └── financial_tools.py                 # @tool decorated functions
│   ├── mock_data.py                            # Seeded demo data
│   └── reasoning.py                           # Local reasoning engine
└── webapp/
    ├── app.py                                  # Flask web app
    └── templates/index.html                    # UI
```

---

## Quick Start

**Run the demo notebook (no AWS account needed):**

```bash
pip install strands-agents
jupyter notebook Multi_Agent_Financial_Advisor_Demo.ipynb
```

**Switch to production (AWS Bedrock):**

```python
# Demo mode
memory = MemoryManager(use_local_fallback=True)
model = LocalReasoningEngine(...)

# Production mode — only these two lines change
memory = MemoryManager(memory_id="<agentcore-memory-id>")
model = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0")
```

**Deploy to AWS:**

```bash
agentcore dev       # local dev with hot-reload
agentcore deploy    # provision serverless endpoint
agentcore invoke    # test your deployed agent
```

---

## Resources

- Companion article: [Building_a_Memory-Powered_Multi-Agent_Financial_Advisor.md](./Building_a_Memory-Powered_Multi-Agent_Financial_Advisor.md)
- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
- [Amazon Bedrock AgentCore Docs](https://docs.aws.amazon.com/bedrock-agentcore)
- [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [AgentCore Samples](https://github.com/awslabs/agentcore-samples)
- [Strands Agents Documentation](https://strandsagents.com/latest)

---

## Prerequisites

- Python 3.10+
- `pip install strands-agents`
- AWS account with Bedrock access (for production mode only)
