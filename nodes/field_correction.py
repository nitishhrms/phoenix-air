"""
Field correction node — handles mid-booking corrections.
Detects which field to change (keyword → LLM), resets downstream fields,
routes back to the correct collection step.
"""

import os
import re
from langchain_core.messages import AIMessage
from services.llm import chat_response

# Removed generic qualifiers ("i meant", "actually", "going to", "arriving", "day")
# — they match too broadly across fields. LLM fallback handles ambiguous cases.
# Removed bare "number" — matches "booking number", "flight number" incorrectly.
_FIELD_KEYWORDS = {
    "departure":   ["departure", "departing", "leaving from", "from city", "origin",
                    "different city"],
    "destination": ["destination", "flying to", "to city", "where i'm going"],
    "date":        ["date", "travel date", "when i'm flying", "different day",
                    "different date", "change the date", "wrong date"],
    "passenger":   ["name", "passenger", "my name", "first name", "last name",
                    "spelled wrong", "wrong name"],
    "contact":     ["contact", "email", "phone number", "phone", "wrong email",
                    "wrong number", "confirmation to", "send to"],
}

_FIELD_STATE_MAP = {
    "departure":   "COLLECTING_DEPARTURE",
    "destination": "COLLECTING_DESTINATION",
    "date":        "COLLECTING_DATE",
    "passenger":   "COLLECTING_PASSENGER",
    "contact":     "COLLECTING_CONTACT",
}

_DOWNSTREAM_CLEARS = {
    "departure":   ["departure_city", "departure_iata", "destination_city", "destination_iata",
                    "travel_date", "travel_dates", "flights", "selected_flight",
                    "passenger_first", "passenger_last", "contact", "contact_type",
                    "confirmation", "payment_confirmed", "transaction_id"],
    "destination": ["destination_city", "destination_iata",
                    "travel_date", "travel_dates", "flights", "selected_flight",
                    "passenger_first", "passenger_last", "contact", "contact_type",
                    "confirmation", "payment_confirmed", "transaction_id"],
    "date":        ["travel_date", "travel_dates", "flights", "selected_flight",
                    "passenger_first", "passenger_last", "contact", "contact_type",
                    "confirmation", "payment_confirmed", "transaction_id"],
    "passenger":   ["passenger_first", "passenger_last", "contact", "contact_type",
                    "confirmation", "payment_confirmed", "transaction_id"],
    "contact":     ["contact", "contact_type", "confirmation",
                    "payment_confirmed", "transaction_id"],
}

_FIELD_RE_ASK = {
    "departure":   "departure city",
    "destination": "destination city",
    "date":        "travel date",
    "passenger":   "full name",
    "contact":     "phone number or email",
}

_CURRENT_STEP_LABELS = {
    "COLLECTING_DEPARTURE":   "departure city",
    "COLLECTING_DESTINATION": "destination city",
    "COLLECTING_DATE":        "travel date",
    "PRESENTING_FLIGHTS":     "flight selection",
    "COLLECTING_PASSENGER":   "passenger name",
    "COLLECTING_CONTACT":     "contact information",
    "COLLECTING_PAYMENT":     "payment details",
}

# Full booking state order — used to block forward-state corrections.
_STATE_ORDER = [
    "IDLE",
    "SELECTING_LANGUAGE",
    "COLLECTING_PHONE",
    "VERIFYING_OTP",
    "COLLECTING_DEPARTURE",
    "COLLECTING_DESTINATION",
    "COLLECTING_DATE",
    "PRESENTING_FLIGHTS",
    "COLLECTING_PASSENGER",
    "COLLECTING_CONTACT",
    "COLLECTING_PAYMENT",
    "BOOKING_CONFIRMED",
    "DONE",
]


def _detect_field_keyword(text: str) -> str | None:
    t = text.lower()
    best_field = None
    best_len = 0
    for field, keywords in _FIELD_KEYWORDS.items():
        for kw in keywords:
            # Multi-word phrases: plain substring match.
            # Single words: word-boundary match to prevent "day" inside "today",
            # "date" inside "update", "name" inside "rename", etc.
            if ' ' in kw:
                matched = kw in t
            else:
                matched = bool(re.search(r'\b' + re.escape(kw) + r'\b', t))
            if matched and len(kw) > best_len:
                best_field = field
                best_len = len(kw)
    return best_field


def _detect_field_llm(text: str) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            system=[
                {
                    "type": "text",
                    "text": (
                        "The user is booking a flight and wants to correct something. "
                        "Identify which field they want to change. "
                        "Reply with EXACTLY one of: departure, destination, date, passenger, contact, unknown."
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": text}],
        )
        result = msg.content[0].text.strip().lower()
        return result if result in _FIELD_STATE_MAP else None
    except Exception:
        return None


def field_correction_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    messages   = state.get("messages", [])
    language   = state.get("language", "en")
    conv_state = state.get("conv_state", "IDLE")

    field = _detect_field_keyword(user_input) or _detect_field_llm(user_input)

    if not field:
        task     = "User wants to change something but it's unclear what. Ask which part to update: departure city, destination, travel date, passenger name, or contact info."
        fallback = "I'd be happy to help! Which part would you like to update — departure city, destination, travel date, passenger name, or contact info?"
        response = chat_response(task, {}, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {**state, "intent": "normal", "response": response, "messages": messages + [msg]}

    # Block forward-state corrections — user hasn't reached that step yet.
    target_state = _FIELD_STATE_MAP[field]
    current_idx  = _STATE_ORDER.index(conv_state) if conv_state in _STATE_ORDER else 0
    target_idx   = _STATE_ORDER.index(target_state)

    if target_idx > current_idx:
        current_step = _CURRENT_STEP_LABELS.get(conv_state, "current information")
        task     = (
            f"The user wants to update their {_FIELD_RE_ASK[field]} but we haven't "
            f"reached that step yet. We are currently collecting the {current_step}. "
            f"Politely explain we'll get to that soon, then ask for their {current_step}."
        )
        fallback = (
            f"We haven't reached the {_FIELD_RE_ASK[field]} step yet — "
            f"let's continue with your {current_step} first!"
        )
        response = chat_response(task, {}, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {**state, "intent": "normal", "response": response, "messages": messages + [msg]}

    patch = {k: None for k in _DOWNSTREAM_CLEARS[field]}
    patch["conv_state"]  = _FIELD_STATE_MAP[field]
    patch["retry_count"] = 0
    patch["intent"]      = "normal"
    patch["suggestions"] = []

    ctx      = {"field_being_corrected": field, "re_ask_for": _FIELD_RE_ASK[field]}
    task     = f"User wants to change their {field}. Acknowledge cheerfully and ask for their {_FIELD_RE_ASK[field]}."
    fallback = f"Of course! What would you like as your {_FIELD_RE_ASK[field]}?"
    response = chat_response(task, ctx, user_input, language=language) or fallback
    msg = AIMessage(content=response)
    patch["response"] = response
    patch["messages"] = messages + [msg]

    return {**state, **patch}
