"""
Security guard node — first node in the graph.
Blocks prompt injection, SQL injection, XSS, and system probes
before any other processing happens.
"""

from langchain_core.messages import AIMessage
from services.security_guard import check_input

_SAFE_RESPONSE = (
    "I'm sorry, I can't process that request. "
    "I'm here to help you book flights with Phoenix Air."
)


def security_guard_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    messages   = state.get("messages", [])

    blocked, reason = check_input(user_input)

    if blocked:
        print(f"[HARDCODED] security_guard blocked: {reason}")
        msg = AIMessage(content=_SAFE_RESPONSE)
        return {
            **state,
            "blocked":       True,
            "response":      _SAFE_RESPONSE,
            "response_type": "hardcoded",
            "messages":      messages + [msg],
        }

    return {**state, "blocked": False}
