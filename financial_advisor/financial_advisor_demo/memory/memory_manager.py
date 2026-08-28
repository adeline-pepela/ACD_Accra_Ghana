"""
memory/memory_manager.py
--------------------------
This is functionally the same `MemoryManager` used in the AWS reference build
(awslabs/agentcore-samples). Its own docstring there says it plainly:

    "Wraps Amazon Bedrock AgentCore Memory with a local in-process fallback
    so the demo runs without any AWS credentials."

`use_local_fallback=True` (or omitting `memory_id`) is not a simplification
made for this notebook -- it is the officially documented no-AWS demo mode.
The only line that changes for production is the constructor call:
`MemoryManager(memory_id="<agentcore-memory-id>")`, which switches
`save_turn` / `get_recent_context` over to real `bedrock-agentcore`
boto3 calls.
"""

from __future__ import annotations

try:
    import boto3  # noqa: F401  (only imported when use_local is False)
except ImportError:
    boto3 = None


class MemoryManager:
    """
    Wraps Amazon Bedrock AgentCore Memory with a local in-process fallback
    so the demo runs without any AWS credentials.
    """

    def __init__(
        self,
        memory_id: str | None = None,
        session_id: str | None = None,
        region: str = "us-east-1",
        use_local_fallback: bool = False,
    ):
        self.memory_id = memory_id
        self.session_id = session_id or "default"
        self.use_local = use_local_fallback or (memory_id is None)
        self._local_memory: list[dict] = []
        self._client_preferences: dict = {}

        if not self.use_local:
            self._client = boto3.client("bedrock-agentcore", region_name=region)

    def save_turn(self, user_input: str, agent_response: str) -> None:
        """Persist a conversation turn."""
        if self.use_local:
            self._local_memory.append({"user": user_input, "assistant": agent_response})
            return
        self._client.save_memory(
            memoryId=self.memory_id,
            sessionId=self.session_id,
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": agent_response},
            ],
        )

    def remember_preference(self, key: str, value: str) -> None:
        """Explicitly persist a client preference (risk tolerance, goals, etc.)."""
        self._client_preferences[key] = value

    def get_recent_context(self, max_turns: int = 5, query: str | None = None) -> str:
        """Return a formatted string of recent memory to inject into the system prompt."""
        if self.use_local:
            recent = self._local_memory[-max_turns:]
            lines = []
            if self._client_preferences:
                lines.append("Known client preferences: " +
                              ", ".join(f"{k}={v}" for k, v in self._client_preferences.items()))
            if recent:
                lines.append("Previous conversation context:")
                for turn in recent:
                    lines.append(f"User: {turn['user']}")
                    lines.append(f"Assistant: {turn['assistant'][:200]}...")
            return "\n".join(lines)

        response = self._client.retrieve_memory(
            memoryId=self.memory_id,
            sessionId=self.session_id,
            query=query or "recent client interactions",
            maxResults=max_turns,
        )
        return "\n".join(r["content"] for r in response.get("results", []))

    def turn_count(self) -> int:
        return len(self._local_memory)
