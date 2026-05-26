"""
Parses the user's flight selection and advances to passenger collection.

Features:
  - Sort detection: "sort by price" / "sort by time" re-sorts flights
  - Self-reflection: verifies the confirmation response names the correct flight
  - All responses generated via Claude
"""

import os
import re
from langchain_core.messages import AIMessage
from services.llm import chat_response, self_reflect_confirmation

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
}

_SORT_PRICE_KWS = ["sort by price", "cheapest", "cheapest first", "by price", "lowest price",
                   "most affordable", "budget", "least expensive", "by cost"]
_SORT_TIME_KWS  = ["sort by time", "fastest", "quickest", "shortest flight", "by duration",
                   "sort by duration", "by time", "quick", "shortest"]


def _detect_sort_intent(text: str) -> str | None:
    t = text.lower()
    if any(k in t for k in _SORT_PRICE_KWS):
        return "price"
    if any(k in t for k in _SORT_TIME_KWS):
        return "time"
    return None


def _sort_flights(flights: list, sort_by: str) -> list:
    copied = [dict(f) for f in flights]
    if sort_by == "price":
        copied.sort(key=lambda f: float(f.get("price", 9999)))
    elif sort_by == "time":
        copied.sort(key=lambda f: int(f.get("durationMinutes", 9999)))
    return copied


def _parse_selection(user_input: str, flights: list) -> int | None:
    text = user_input.lower().strip()

    m = re.search(r'\b(\d+)\b', text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(flights):
            return idx

    for word, num in ORDINALS.items():
        if word in text:
            idx = num - 1
            if 0 <= idx < len(flights):
                return idx

    for i, f in enumerate(flights):
        airline = f.get("airline", "").lower()
        fn      = f.get("flightNumber", "").lower()
        if airline in text or fn in text or text in airline:
            return i

    return None


def present_flights_node(state: dict) -> dict:
    user_input  = state.get("user_input", "")
    flights     = state.get("flights") or []
    messages    = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    language    = state.get("language", "en")

    # ── Sort detection (handle before selection parsing) ───────────
    sort_by = _detect_sort_intent(user_input)
    if sort_by:
        sorted_flights = _sort_flights(flights, sort_by)
        sort_label = "price (cheapest first)" if sort_by == "price" else "duration (fastest first)"
        ctx      = {"sort_type": sort_label, "flight_count": len(sorted_flights)}
        task     = f"User wants flights sorted by {sort_label}. Acknowledge and invite them to pick one."
        fallback = f"Sure! Here are the flights sorted by {sort_label}. Which one would you like?"
        response = chat_response(task, ctx, user_input, language=language, skip_hallucination_guard=False) or fallback
        msg = AIMessage(content=response)
        return {
            **state,
            "flights":     sorted_flights,
            "flight_sort": sort_by,
            "response":    response,
            "messages":    messages + [msg],
        }

    # ── Flight selection ───────────────────────────────────────────
    idx = _parse_selection(user_input, flights)

    if idx is None:
        retry_count += 1
        options  = ", ".join(f"option {i+1} ({f.get('airline')})" for i, f in enumerate(flights))
        ctx      = {"available_options": options}
        task     = f"User's selection wasn't understood. List options ({options}) and ask which they prefer."
        fallback = f"I didn't catch that. You can say {options}. Which would you prefer?"
        response = chat_response(task, ctx, user_input, language=language, skip_hallucination_guard=False) or fallback
        msg = AIMessage(content=response)
        return {**state, "response": response, "retry_count": retry_count, "messages": messages + [msg]}

    selected = flights[idx]
    ctx = {
        "selected_airline":  selected.get("airline"),
        "flight_number":     selected.get("flightNumber"),
        "departure_time":    selected.get("departureTime"),
        "arrival_time":      selected.get("arrivalTime"),
        "price":             selected.get("price"),
    }
    task     = (
        f"User selected {selected.get('airline')} flight {selected.get('flightNumber')}. "
        "Confirm their selection enthusiastically and ask for their full name for the booking."
    )
    fallback = (
        f"Great choice! {selected.get('airline')} flight {selected.get('flightNumber')}. "
        "Could I get your full name for the booking?"
    )
    response = chat_response(task, ctx, user_input, language=language, skip_hallucination_guard=False) or fallback

    # ── Self-reflection: verify response mentions the correct flight ─
    if not self_reflect_confirmation(selected, response):
        # Regenerate with a more explicit instruction
        task_retry = (
            f"Confirm EXPLICITLY that the user selected {selected.get('airline')} "
            f"flight {selected.get('flightNumber')}. Then ask for their full name."
        )
        response = chat_response(task_retry, ctx, user_input, language=language) or fallback

    msg = AIMessage(content=response)
    return {
        **state,
        "conv_state":      "COLLECTING_PASSENGER",
        "selected_flight": selected,
        "retry_count":     0,
        "flight_sort":     None,
        "response":        response,
        "messages":        messages + [msg],
    }
