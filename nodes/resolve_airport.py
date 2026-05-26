"""
Resolves user city input to an IATA code — 4-layer pipeline:

  Layer 1 (rule-based)   : Exact match on IATA code, city name, aliases in SQLite
  Layer 2 (rule-based)   : SequenceMatcher fuzzy — catches common typos
  Layer 3 (embedding)    : Cosine similarity on city name vectors — catches
                           semantic variants ("Chi-town", "Big Apple", "Silicon Valley")
  Layer 4 (LLM)          : Haiku city validator — checks if input is a real city
                           with a major airport; distinguishes "not in our network"
                           from "city does not exist"

Handles both COLLECTING_DEPARTURE and COLLECTING_DESTINATION states.
Policy / general / OOD queries are still allowed (handled upstream by intent_router).
"""

import numpy as np
from db.database import resolve_airport, suggest_airports, get_conn
from langchain_core.messages import AIMessage
from services.llm import chat_response
from services.city_validator import llm_check_city

MAX_RETRIES         = 3
EMBED_THRESHOLD     = 0.55   # minimum cosine similarity to trust embedding match
FUZZY_THRESHOLD     = 0.45   # SequenceMatcher ratio — already used in database.py

# Hardcoded responses (no LLM — clear, predictable)
_NO_CITY_FOUND = (
    "I couldn't find any city or airport matching \"{input}\". "
    "Please try a major city name, e.g. New York, London, or Tokyo."
)

# ── Lazy embedding helpers ─────────────────────────────────────────────────────

from services.embedder import get_embedder as _get_embedder

_city_vectors  = {}   # iata → (city, vector)


def _get_city_vectors() -> dict:
    """Build/return cached city-name embeddings for every airport in the DB."""
    global _city_vectors
    if _city_vectors:
        return _city_vectors

    conn   = get_conn()
    rows   = conn.execute("SELECT iata, city, name FROM airports").fetchall()
    conn.close()

    if not rows:
        return {}

    embedder = _get_embedder()
    if embedder is None:
        return {}
    cities  = [r["city"] for r in rows]
    vectors = embedder.encode(cities, normalize_embeddings=True)

    for row, vec in zip(rows, vectors):
        _city_vectors[row["iata"]] = {
            "city":   row["city"],
            "vector": vec,
        }

    print(f"[EMBED] Loaded vectors for {len(_city_vectors)} airports")
    return _city_vectors


def _embedding_suggest(query: str, top_k: int = 3) -> list[dict]:
    """Layer 3: cosine-similarity match against all city name embeddings."""
    try:
        embedder = _get_embedder()
        if embedder is None:
            return []
        q_vec = embedder.encode([query], normalize_embeddings=True)[0]
        vectors  = _get_city_vectors()

        scored = []
        for iata, data in vectors.items():
            score = float(np.dot(q_vec, data["vector"]))
            if score >= EMBED_THRESHOLD:
                scored.append((score, iata, data["city"]))

        scored.sort(reverse=True)
        return [
            {"iata": iata, "city": city, "display": f"{city} ({iata})", "score": score}
            for score, iata, city in scored[:top_k]
        ]
    except Exception as e:
        print(f"[EMBED] suggest error: {e}")
        return []


# ── Main node ──────────────────────────────────────────────────────────────────

def resolve_airport_node(state: dict) -> dict:
    conv_state  = state.get("conv_state", "IDLE")
    user_input  = state.get("user_input", "")
    retry_count = state.get("retry_count", 0)
    intent      = state.get("intent", "normal")
    messages    = state.get("messages", [])
    language    = state.get("language", "en")

    # ── Restart: wipe state, begin fresh ──────────────────────────────────────
    if intent == "restart" or conv_state == "IDLE":
        task     = "The user wants to start a new booking. Acknowledge briefly and ask for their departure city."
        fallback = "Sure! Let's start fresh. Which city are you departing from?"
        response = chat_response(task, {}, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {
            **state,
            "conv_state":       "COLLECTING_DEPARTURE",
            "departure_city":   None, "departure_iata":   None,
            "destination_city": None, "destination_iata": None,
            "travel_date":      None, "flights":          None,
            "selected_flight":  None, "passenger_first":  None,
            "passenger_last":   None, "contact":          None,
            "contact_type":     None, "confirmation":     None,
            "retry_count":      0,    "intent":           "normal",
            "suggestions":      [],   "response":         response,
            "messages":         messages + [msg],
        }

    which = "departure" if conv_state == "COLLECTING_DEPARTURE" else "destination"

    # ── Layer 1: exact DB match ────────────────────────────────────────────────
    result = resolve_airport(user_input)

    if result:
        iata = result["iata"]
        city = result["city"]
        print(f"[RESOLVE L1] rule-based match: {user_input!r} -> {city} ({iata})")
        return _confirmed(state, conv_state, city, iata, user_input, messages, language)

    # ── Layer 2: SequenceMatcher fuzzy ────────────────────────────────────────
    fuzzy_hints = suggest_airports(user_input, top_k=3)
    if fuzzy_hints:
        print(f"[RESOLVE L2] fuzzy match for {user_input!r}: {[h['city'] for h in fuzzy_hints]}")
        return _ask_suggestions(state, user_input, fuzzy_hints, which, retry_count, messages, language)

    # ── Layer 3: embedding similarity ─────────────────────────────────────────
    embed_hints = _embedding_suggest(user_input, top_k=3)
    if embed_hints:
        print(f"[RESOLVE L3] embedding match for {user_input!r}: {[h['city'] for h in embed_hints]}")
        return _ask_suggestions(state, user_input, embed_hints, which, retry_count, messages, language)

    # ── Layer 4: LLM city validator ───────────────────────────────────────────
    print(f"[RESOLVE L4] LLM city check for {user_input!r}")
    has_airport, corrected = llm_check_city(user_input)

    if has_airport:
        # Real city but not in our network — try DB again with corrected name, then suggest nearby
        db_result = resolve_airport(corrected) if corrected else None
        if db_result:
            # Corrected name matched — use it
            print(f"[RESOLVE L4] corrected '{user_input}' → '{corrected}' found in DB")
            return _confirmed(state, conv_state, db_result["city"], db_result["iata"],
                              user_input, messages, language)

        # City is real but we don't serve it — give nearby suggestions
        nearby = suggest_airports(corrected or user_input, top_k=3)
        city_display = corrected or user_input
        if nearby:
            ctx  = {"searched_city": city_display, "nearby_options": ", ".join(h["city"] for h in nearby)}
            task = (
                f"The user wants to fly from/to {city_display} but we don't serve it directly. "
                f"Suggest these nearby airports we do serve: {ctx['nearby_options']}. "
                "Ask which they'd prefer."
            )
            fallback = (
                f"We don't currently fly directly to {city_display}, "
                f"but we do serve nearby: {ctx['nearby_options']}. Which would work for you?"
            )
        else:
            ctx  = {"searched_city": city_display}
            task = (
                f"We don't serve {city_display}. "
                "Apologise and ask the user to name another major city we might serve."
            )
            fallback = (
                f"Unfortunately we don't serve {city_display} directly. "
                "Could you name another nearby city?"
            )
        response = chat_response(task, ctx, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {
            **state,
            "retry_count": retry_count + 1,
            "suggestions": nearby,
            "response":    response,
            "messages":    messages + [msg],
        }

    # City does not exist → hardcoded response
    print(f"[RESOLVE L4] city not found: {user_input!r}")
    no_city_msg = _NO_CITY_FOUND.format(input=user_input)
    msg = AIMessage(content=no_city_msg)
    return {
        **state,
        "retry_count":   retry_count + 1,
        "suggestions":   [],
        "response":      no_city_msg,
        "response_type": "hardcoded",
        "messages":      messages + [msg],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _confirmed(state, conv_state, city, iata, user_input, messages, language) -> dict:
    """City resolved — advance conv_state and ask for next field."""
    if conv_state == "COLLECTING_DEPARTURE":
        ctx      = {"departure_city": city, "departure_iata": iata}
        task     = f"Departure city confirmed as {city} ({iata}). Ask for the destination city."
        fallback = f"Got it, departing from {city} ({iata}). Where are you flying to?"
        response = chat_response(task, ctx, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {
            **state,
            "conv_state":     "COLLECTING_DESTINATION",
            "departure_city": city, "departure_iata": iata,
            "retry_count":    0,   "suggestions":    [],
            "response":       response, "messages": messages + [msg],
        }

    if conv_state == "COLLECTING_DESTINATION":
        dep_iata = state.get("departure_iata", "")
        if iata == dep_iata:
            task     = "Same city chosen for departure and destination. Politely point that out and ask again."
            fallback = "Departure and destination can't be the same city. Where would you like to fly to?"
            response = chat_response(task, {"departure_city": state.get("departure_city")},
                                     user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {**state, "suggestions": [], "response": response, "messages": messages + [msg]}

        ctx = {
            "departure_city":   state.get("departure_city"),
            "destination_city": city,
            "destination_iata": iata,
        }
        _fmt_hint = "Please enter it as: Month Day Year (e.g. August 15 2026)."
        task     = (
            f"Route confirmed: {state.get('departure_city')} to {city} ({iata}). "
            f"Ask for travel date. Always end your reply with: '{_fmt_hint}'"
        )
        fallback = (
            f"Great, flying from {state.get('departure_city')} to {city} ({iata}). "
            f"What date would you like to travel? {_fmt_hint}"
        )
        response = chat_response(task, ctx, user_input, language=language) or fallback
        # Guarantee the format hint appears even if the LLM omits it
        if "august" not in response.lower() and "e.g." not in response.lower() and "for example" not in response.lower():
            response = response.rstrip() + f" {_fmt_hint}"
        msg = AIMessage(content=response)
        return {
            **state,
            "conv_state":       "COLLECTING_DATE",
            "destination_city": city, "destination_iata": iata,
            "retry_count":      0,   "suggestions":       [],
            "response":         response, "messages": messages + [msg],
        }

    # Fallback (shouldn't reach here)
    return state


def _ask_suggestions(state, user_input, hints, which, retry_count, messages, language) -> dict:
    """Layers 2 & 3: present fuzzy/embedding suggestions and ask user to confirm."""
    options  = ", ".join(h["city"] for h in hints)
    ctx      = {"searched_city": user_input, "which_field": which, "suggestions": options}
    task     = f"City not found for '{user_input}'. Suggest these alternatives: {options}. Ask which they meant."
    fallback = f"I couldn't find \"{user_input}\". Did you mean one of: {options}?"
    response = chat_response(task, ctx, user_input, language=language) or fallback
    msg = AIMessage(content=response)
    return {
        **state,
        "retry_count": retry_count + 1,
        "suggestions": hints,
        "response":    response,
        "messages":    messages + [msg],
    }
