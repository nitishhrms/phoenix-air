from langchain_core.messages import AIMessage
from services.llm import chat_response

_STATE_LABELS = {
    "COLLECTING_DEPARTURE":   "waiting for departure city",
    "COLLECTING_DESTINATION": "waiting for destination city",
    "COLLECTING_DATE":        "waiting for travel date",
    "PRESENTING_FLIGHTS":     "showing flight options",
    "COLLECTING_PASSENGER":   "waiting for passenger name",
    "COLLECTING_CONTACT":     "waiting for contact info",
    "COLLECTING_PAYMENT":     "waiting for payment confirmation",
    "POST_BOOKING":           "booking completed",
}


def greeting_node(state: dict) -> dict:
    conv_state = state.get("conv_state", "IDLE")
    messages   = state.get("messages", [])
    language   = state.get("language", "en")

    ctx = {
        "booking_step": _STATE_LABELS.get(conv_state, "just starting"),
        "departure":    state.get("departure_city"),
        "destination":  state.get("destination_city"),
        "travel_date":  state.get("travel_date"),
    }

    if conv_state in ("IDLE", "SELECTING_LANGUAGE", "COLLECTING_PHONE", "VERIFYING_OTP"):
        task    = "Greet the user warmly and let them know you can help them book a flight."
        fallback = "Hello! Welcome to Phoenix Air. I can help you book a flight — where are you departing from?"
    elif conv_state == "POST_BOOKING":
        task    = (
            "The user's booking is complete. Ask warmly if there is anything else you can help them with — "
            "such as policy questions, baggage info, or any other travel queries."
        )
        fallback = "Is there anything else I can help you with today?"
    else:
        task    = (
            f"The user said hello mid-booking. Greet them briefly and naturally "
            f"re-state where we left off ({_STATE_LABELS.get(conv_state, 'the booking')}) "
            "so they know how to continue."
        )
        fallback = "Hi there! Let's keep going with your booking — how can I help?"

    response = chat_response(task, ctx, state.get("user_input", ""), language=language) or fallback
    msg = AIMessage(content=response)
    return {**state, "intent": "normal", "response": response, "messages": messages + [msg]}
