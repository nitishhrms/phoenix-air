"""
Detects contact type, calls external booking API,
saves booking to SQLite, sends mock notification.
"""

import random
import string
from langchain_core.messages import AIMessage
from services.external_api import book_flight as api_book_flight
from services.notifications import detect_contact_type, send_confirmation
from services.llm import chat_response
from db.database import save_booking, upsert_user_profile


def _gen_confirmation() -> str:
    chars = string.ascii_uppercase + string.digits
    return "PHN-" + "".join(random.choices(chars, k=6))


def book_flight_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    messages   = state.get("messages", [])
    language   = state.get("language", "en")
    contact    = user_input.strip()

    contact_type = detect_contact_type(contact)

    flight     = state.get("selected_flight", {})
    src        = state.get("departure_iata", "")
    dst        = state.get("destination_iata", "")
    date       = state.get("travel_date", "")
    first      = state.get("passenger_first", "")
    last       = state.get("passenger_last", "")
    session_id = state.get("session_id", "")

    api_result = api_book_flight(
        src=src, dst=dst, date=date,
        flight_id=flight.get("flightId", ""),
        first_name=first, last_name=last,
    )

    confirmation = (
        api_result.get("confirmationNumber")
        or api_result.get("confirmation_number")
        or api_result.get("bookingId")
        or _gen_confirmation()
    )

    booking_record = {
        "confirmation_number": confirmation,
        "session_id":          session_id,
        "flight_id":           flight.get("flightId", ""),
        "airline":             flight.get("airline", ""),
        "flight_number":       flight.get("flightNumber", ""),
        "departure_time":      flight.get("departureTime", ""),
        "arrival_time":        flight.get("arrivalTime", ""),
        "src_iata":            src,
        "dst_iata":            dst,
        "passenger_name":      f"{first} {last}",
        "contact":             contact,
        "contact_type":        contact_type,
    }
    save_booking(booking_record)

    caller_phone = state.get("caller_phone", "")
    if caller_phone:
        upsert_user_profile(caller_phone, first, last, src, dst)

    send_confirmation(contact, contact_type, booking_record)

    price = float(flight.get("price", 0.0))
    ctx   = {
        "passenger_first": first,
        "contact":         contact,
        "total_price":     f"${price:.2f}",
    }
    task     = (
        f"Contact info collected ({contact}). Tell {first} their total is ${price:.2f} "
        "and ask them to confirm payment with 'yes'."
    )
    fallback = (
        f"Thanks, {first}! Your total is ${price:.2f}. "
        "Shall I proceed with payment? Say yes to confirm."
    )
    response = chat_response(task, ctx, user_input, language=language) or fallback
    msg = AIMessage(content=response)

    return {
        **state,
        "conv_state":   "COLLECTING_PAYMENT",
        "contact":      contact,
        "contact_type": contact_type,
        "confirmation": confirmation,
        "flight_price": price,
        "end_call":     False,
        "response":     response,
        "messages":     messages + [msg],
    }
