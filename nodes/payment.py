"""
Payment node — confirms charge, processes mock payment,
generates PDF ticket, sends it, then moves to BOOKING_CONFIRMED.
"""

from langchain_core.messages import AIMessage
from services.payment import mock_charge
from services.pdf_ticket import generate_ticket, send_ticket
from services.llm import chat_response

_YES = ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed", "go ahead", "please"]
_NO  = ["no", "nope", "cancel", "stop", "don't", "do not", "back"]


def payment_node(state: dict) -> dict:
    user_input   = state.get("user_input", "").lower().strip()
    messages     = state.get("messages", [])
    retry_count  = state.get("retry_count", 0)
    language     = state.get("language", "en")
    flight       = state.get("selected_flight", {}) or {}
    price        = float(flight.get("price", 0.0))
    contact      = state.get("contact", "")
    contact_type = state.get("contact_type", "email")
    first        = state.get("passenger_first", "")
    confirmation = state.get("confirmation", "")

    # ── First entry or ambiguous — show total and ask ──────────────────
    if not user_input or not any(w in user_input for w in _YES + _NO):
        ctx  = {
            "airline":       flight.get("airline"),
            "flight_number": flight.get("flightNumber"),
            "total_price":   f"${price:.2f}",
            "contact":       contact,
        }
        task     = (
            f"Show the payment summary: {flight.get('airline')} flight {flight.get('flightNumber')}, "
            f"total ${price:.2f}, confirmation to {contact}. Ask them to say yes to confirm."
        )
        fallback = (
            f"Your total is ${price:.2f} for {flight.get('airline','')} flight {flight.get('flightNumber','')}. "
            f"Shall I process payment to {contact}? Say yes to confirm."
        )
        response = chat_response(task, ctx, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {**state, "flight_price": price, "response": response, "messages": messages + [msg]}

    # ── User declined ──────────────────────────────────────────────────
    if any(w in user_input for w in _NO):
        retry_count += 1
        if retry_count >= 2:
            task     = "User declined payment twice. Politely cancel and say goodbye."
            fallback = "No problem. Your session has been cancelled. Goodbye!"
            response = chat_response(task, {}, user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {**state, "end_call": True, "retry_count": retry_count,
                    "response": response, "messages": messages + [msg]}
        task     = "User said no to payment. Let them know they can say yes when ready or cancel."
        fallback = "Okay, let me know when you're ready to proceed, or say cancel to end."
        response = chat_response(task, {}, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {**state, "retry_count": retry_count, "response": response, "messages": messages + [msg]}

    # ── User confirmed — process payment ──────────────────────────────
    result = mock_charge(price, contact, confirmation)
    txn_id = result.get("transaction_id", "")

    booking_record = {
        "confirmation_number": confirmation,
        "passenger_name":      f"{state.get('passenger_first','')} {state.get('passenger_last','')}",
        "airline":             flight.get("airline", ""),
        "flight_number":       flight.get("flightNumber", ""),
        "src_iata":            state.get("departure_iata", ""),
        "dst_iata":            state.get("destination_iata", ""),
        "departure_time":      flight.get("departureTime", ""),
        "arrival_time":        flight.get("arrivalTime", ""),
    }

    pdf_bytes = generate_ticket(booking_record)
    send_ticket(contact, contact_type, pdf_bytes, booking_record)

    method = "SMS" if contact_type == "phone" else "email"
    ctx    = {
        "passenger_first":     first,
        "total_paid":          f"${price:.2f}",
        "transaction_id":      txn_id,
        "confirmation_number": confirmation,
        "airline":             flight.get("airline"),
        "flight_number":       flight.get("flightNumber"),
        "departure_city":      state.get("departure_city"),
        "destination_city":    state.get("destination_city"),
        "travel_date":         state.get("travel_date"),
        "ticket_sent_to":      contact,
        "delivery_method":     method,
    }
    task = (
        f"Payment of ${price:.2f} was successful (transaction {txn_id}). "
        f"Ticket sent to {contact} by {method}. Confirmation: {confirmation}. "
        "Give a warm, complete booking summary and thank them for choosing Phoenix Air."
    )
    fallback = (
        f"Payment of ${price:.2f} confirmed! Transaction: {txn_id}. "
        f"Ticket sent to {contact} by {method}. Confirmation: {confirmation}. "
        f"You're flying {flight.get('airline','')} {flight.get('flightNumber','')} "
        f"from {state.get('departure_city','')} to {state.get('destination_city','')} "
        f"on {state.get('travel_date','')}. Thank you for choosing Phoenix Air!"
    )
    response = chat_response(task, ctx, user_input, max_tokens=200, language=language) or fallback
    msg = AIMessage(content=response)
    return {
        **state,
        "conv_state":        "POST_BOOKING",
        "payment_confirmed": True,
        "transaction_id":    txn_id,
        "end_call":          False,
        "response":          response,
        "messages":          messages + [msg],
    }
