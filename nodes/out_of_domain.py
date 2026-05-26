from langchain_core.messages import AIMessage

_FIXED_RESPONSE = (
    "I specialise in airline booking and can help you search for flights, "
    "book a trip, or answer Phoenix Air policy questions. "
    "Is there something along those lines I can help with?"
)


def out_of_domain_node(state: dict) -> dict:
    print("[HARDCODED] out_of_domain: fixed response")
    messages = state.get("messages", [])
    msg = AIMessage(content=_FIXED_RESPONSE)
    return {
        **state,
        "response":      _FIXED_RESPONSE,
        "response_type": "hardcoded",
        "intent":        "normal",
        "messages":      messages + [msg],
    }
