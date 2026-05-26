"""Parses the passenger's full name and advances to contact collection."""

from langchain_core.messages import AIMessage
from services.llm import chat_response


def _parse_name(text: str) -> tuple[str, str] | None:
    parts = text.strip().split()
    if len(parts) < 2:
        return None
    first = parts[0].capitalize()
    last  = " ".join(p.capitalize() for p in parts[1:])
    return first, last


def collect_passenger_node(state: dict) -> dict:
    user_input  = state.get("user_input", "")
    messages    = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    language    = state.get("language", "en")

    result = _parse_name(user_input)

    if not result:
        retry_count += 1
        task     = "The user didn't provide a full name. Politely ask for both first and last name."
        fallback = "I need your full name — please say both your first and last name."
        response = chat_response(task, {}, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {**state, "response": response, "retry_count": retry_count, "messages": messages + [msg]}

    first, last = result
    ctx      = {"passenger_name": f"{first} {last}"}
    task     = f"Passenger name recorded as {first} {last}. Ask for phone number or email for the confirmation."
    fallback = f"Thank you, {first} {last}. What phone number or email should I send your confirmation to?"
    response = chat_response(task, ctx, user_input, language=language) or fallback
    msg = AIMessage(content=response)
    return {
        **state,
        "conv_state":      "COLLECTING_CONTACT",
        "passenger_first": first,
        "passenger_last":  last,
        "retry_count":     0,
        "response":        response,
        "messages":        messages + [msg],
    }
