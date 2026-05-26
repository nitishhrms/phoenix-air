from langchain_core.messages import AIMessage
from services.llm import chat_response

_TRANSFER_TRIGGERS = ["yes", "yeah", "please", "transfer", "agent", "human", "sure", "ok", "yep"]

_SUPPORT_CTX = {
    "support_email":  "support@phoenixair.com",
    "support_phone":  "1-800-PHX-AIRL (1-800-749-2475)",
    "baggage_portal": "phoenixair.com/baggage",
    "availability":   "24/7",
}

_FALLBACK_INFO     = (
    "Phoenix Air Customer Support is available 24/7 at support@phoenixair.com "
    "or 1-800-749-2475. Would you like me to transfer you to a live agent?"
)
_FALLBACK_TRANSFER = "Connecting you to a live agent now. Please hold."


def customer_contact_node(state: dict) -> dict:
    user_input = state.get("user_input", "").lower()
    messages   = state.get("messages", [])
    language   = state.get("language", "en")

    if any(t in user_input for t in _TRANSFER_TRIGGERS):
        task     = "The user confirmed transfer. Let them know you're connecting them now."
        response = chat_response(task, _SUPPORT_CTX, user_input, language=language) or _FALLBACK_TRANSFER
        msg = AIMessage(content=response)
        return {
            **state,
            "intent":   "normal",
            "transfer": True,
            "end_call": True,
            "response": response,
            "messages": messages + [msg],
        }

    task = (
        "The user has a complaint or support question. "
        "Acknowledge empathetically, share the support details from context, "
        "and ask if they'd like to be transferred to a live agent."
    )
    response = chat_response(task, _SUPPORT_CTX, user_input, language=language) or _FALLBACK_INFO
    msg = AIMessage(content=response)
    return {
        **state,
        "intent":   "normal",
        "response": response,
        "messages": messages + [msg],
    }
