from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AirlineState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────
    session_id:        str
    caller_phone:      Optional[str]
    user_id:           Optional[int]   # web-login account ID (skips OTP if set)

    # ── Language ──────────────────────────────────────────────────
    language:          str             # en | es | fr | hi | zh | ar | pt | de

    # ── Authentication ────────────────────────────────────────────
    authenticated:     bool
    auth_phone:        Optional[str]

    # ── Conversation ──────────────────────────────────────────────
    conv_state:        str
    messages:          Annotated[list, add_messages]
    user_input:        str
    response:          str

    # ── Booking data (filled step by step) ────────────────────────
    departure_city:    Optional[str]
    departure_iata:    Optional[str]
    destination_city:  Optional[str]
    destination_iata:  Optional[str]
    travel_date:       Optional[str]        # YYYY-MM-DD (primary)
    travel_dates:      Optional[list]       # list of YYYY-MM-DD for multi-date search
    flights:           Optional[list]       # raw API flight objects
    selected_flight:   Optional[dict]
    passenger_first:   Optional[str]
    passenger_last:    Optional[str]
    contact:           Optional[str]
    contact_type:      Optional[str]        # "phone" | "email"
    confirmation:      Optional[str]        # PHN-XXXXXX

    # ── Payment ───────────────────────────────────────────────────
    flight_price:      Optional[float]
    payment_confirmed: bool
    transaction_id:    Optional[str]

    # ── Routing / control ─────────────────────────────────────────
    intent:            str
    retry_count:       int
    end_call:          bool
    transfer:          bool
    blocked:           bool
    error:             Optional[str]
    suggestions:       Optional[list]
    flight_sort:       Optional[str]        # "price" | "time" | ""
    cot_analysis:      Optional[dict]       # CoT flight analysis result
    response_type:     Optional[str]        # "llm" | "hardcoded" | "fallback"
    input_type:        Optional[str]        # "answer" | "question" | "query" (from input_classifier)
    validated:         Optional[bool]       # set by answer_validator: True = proceed, False = ask user
    pending_correction: Optional[str]       # LLM-suggested correction waiting for user confirm


CONV_STATES = [
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


def empty_state(session_id: str, caller_phone: str = "", language: str = "en",
                user_id: int = None) -> dict:
    return {
        "session_id":        session_id,
        "caller_phone":      caller_phone,
        "user_id":           user_id,
        "language":          language,
        # auth
        "authenticated":     False,
        "auth_phone":        None,
        # conversation
        "conv_state":        "SELECTING_LANGUAGE",
        "messages":          [],
        "user_input":        "",
        "response":          "",
        # booking
        "departure_city":    None,
        "departure_iata":    None,
        "destination_city":  None,
        "destination_iata":  None,
        "travel_date":       None,
        "travel_dates":      None,
        "flights":           None,
        "selected_flight":   None,
        "passenger_first":   None,
        "passenger_last":    None,
        "contact":           None,
        "contact_type":      None,
        "confirmation":      None,
        # payment
        "flight_price":      None,
        "payment_confirmed": False,
        "transaction_id":    None,
        # routing
        "intent":            "normal",
        "retry_count":       0,
        "end_call":          False,
        "transfer":          False,
        "blocked":           False,
        "error":             None,
        "suggestions":       [],
        "flight_sort":       None,
        "cot_analysis":        None,
        "response_type":       "llm",
        "input_type":          None,
        "validated":           None,
        "pending_correction":  None,
    }
