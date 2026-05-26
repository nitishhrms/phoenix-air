from langchain_core.messages import AIMessage
from services.llm import chat_response

_CTX = {
    "support_phone": "1-800-749-2475",
    "support_email": "support@phoenixair.com",
}
_FALLBACK = "Of course! Connecting you to a live agent now. Please hold for a moment."


def transfer_node(state: dict) -> dict:
    language = state.get("language", "en")
    messages = state.get("messages", [])

    task = (
        "The user wants to speak to a human agent. "
        "Let them know you're connecting them right away and they should hold briefly."
    )
    response = chat_response(task, _CTX, state.get("user_input", ""), language=language) or _FALLBACK
    msg = AIMessage(content=response)
    return {
        **state,
        "response":  response,
        "transfer":  True,
        "end_call":  True,
        "messages":  messages + [msg],
    }
