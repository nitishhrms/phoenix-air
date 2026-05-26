from langgraph.graph import StateGraph, END
from state import AirlineState

from nodes.security_guard    import security_guard_node
from nodes.intent_router     import intent_router_node
from nodes.language_selection import language_selection_node
from nodes.resolve_airport   import resolve_airport_node
from nodes.search_flights    import search_flights_node
from nodes.present_flights   import present_flights_node
from nodes.collect_passenger import collect_passenger_node
from nodes.book_flight       import book_flight_node
from nodes.rag_policy        import rag_policy_node
from nodes.transfer          import transfer_node
from nodes.out_of_domain     import out_of_domain_node
from nodes.greeting          import greeting_node
from nodes.customer_contact  import customer_contact_node
from nodes.field_correction  import field_correction_node
from nodes.auth              import auth_node
from nodes.payment           import payment_node
from nodes.input_classifier  import input_classifier_node
from nodes.answer_validator  import answer_validator_node


def should_continue(state: dict) -> str:
    """
    Conditional edge — reads intent (Layer 1) then conv_state (Layer 2).
    Returns the name of the next node to run.
    """
    intent     = state.get("intent", "normal")
    conv_state = state.get("conv_state", "IDLE")

    # Layer 1: global interruptions
    if intent == "transfer":
        return "transfer"
    if intent == "policy":
        return "rag_policy"
    if intent == "restart":
        return "resolve_airport"
    if intent == "out_of_domain":
        return "out_of_domain"
    if intent == "greeting":
        return "greeting"
    if intent == "field_correction":
        return "field_correction"
    if intent == "customer_contact":
        return "customer_contact"
    if intent == "general_qa":
        return "rag_policy"

    # Layer 2: state-driven routing
    routing = {
        "SELECTING_LANGUAGE":      "language_selection",
        "IDLE":                    "resolve_airport",
        "COLLECTING_PHONE":        "auth",
        "VERIFYING_OTP":           "auth",
        "COLLECTING_DEPARTURE":    "resolve_airport",
        "COLLECTING_DESTINATION":  "resolve_airport",
        "COLLECTING_DATE":         "search_flights",
        "PRESENTING_FLIGHTS":      "present_flights",
        "COLLECTING_PASSENGER":    "collect_passenger",
        "COLLECTING_CONTACT":      "book_flight",
        "COLLECTING_PAYMENT":      "payment",
        "POST_BOOKING":            "greeting",
        "BOOKING_CONFIRMED":       "end_node",
        "DONE":                    "end_node",
    }

    return routing.get(conv_state, "end_node")


def after_security(state: dict) -> str:
    return "end_node" if state.get("blocked") else "input_classifier"


def after_classifier(state: dict) -> str:
    """Route to answer_validator for answers, directly to intent_router for questions/queries."""
    return "answer_validator" if state.get("input_type") == "answer" else "intent_router"


def after_validator(state: dict) -> str:
    """Continue to intent_router if answer is valid, stop and return response if not."""
    return "intent_router" if state.get("validated") else END


def after_rag_policy(state: dict) -> str:
    return "customer_contact" if state.get("intent") == "customer_contact" else END


def end_node(state: dict) -> dict:
    return {**state, "end_call": True}


def build_graph():
    graph = StateGraph(AirlineState)

    graph.add_node("security_guard",    security_guard_node)
    graph.add_node("input_classifier",  input_classifier_node)
    graph.add_node("answer_validator",  answer_validator_node)
    graph.add_node("intent_router",     intent_router_node)
    graph.add_node("language_selection", language_selection_node)
    graph.add_node("resolve_airport",   resolve_airport_node)
    graph.add_node("search_flights",    search_flights_node)
    graph.add_node("present_flights",   present_flights_node)
    graph.add_node("collect_passenger", collect_passenger_node)
    graph.add_node("book_flight",       book_flight_node)
    graph.add_node("rag_policy",        rag_policy_node)
    graph.add_node("transfer",          transfer_node)
    graph.add_node("end_node",          end_node)
    graph.add_node("out_of_domain",     out_of_domain_node)
    graph.add_node("greeting",          greeting_node)
    graph.add_node("customer_contact",  customer_contact_node)
    graph.add_node("field_correction",  field_correction_node)
    graph.add_node("auth",              auth_node)
    graph.add_node("payment",           payment_node)

    graph.set_entry_point("security_guard")
    graph.add_conditional_edges(
        "security_guard",
        after_security,
        {"end_node": "end_node", "input_classifier": "input_classifier"},
    )

    graph.add_conditional_edges(
        "input_classifier",
        after_classifier,
        {"answer_validator": "answer_validator", "intent_router": "intent_router"},
    )

    graph.add_conditional_edges(
        "answer_validator",
        after_validator,
        {"intent_router": "intent_router", END: END},
    )

    graph.add_conditional_edges(
        "intent_router",
        should_continue,
        {
            "language_selection": "language_selection",
            "transfer":           "transfer",
            "rag_policy":         "rag_policy",
            "resolve_airport":    "resolve_airport",
            "search_flights":     "search_flights",
            "present_flights":    "present_flights",
            "collect_passenger":  "collect_passenger",
            "book_flight":        "book_flight",
            "end_node":           "end_node",
            "out_of_domain":      "out_of_domain",
            "greeting":           "greeting",
            "customer_contact":   "customer_contact",
            "field_correction":   "field_correction",
            "auth":               "auth",
            "payment":            "payment",
            "general_qa":         "rag_policy",
        },
    )

    graph.add_conditional_edges(
        "rag_policy",
        after_rag_policy,
        {"customer_contact": "customer_contact", END: END},
    )

    for node in [
        "language_selection", "resolve_airport", "search_flights", "present_flights",
        "collect_passenger", "book_flight", "transfer",
        "end_node", "out_of_domain", "greeting", "customer_contact",
        "field_correction", "auth", "payment",
    ]:
        graph.add_edge(node, END)

    return graph.compile()


airline_graph = build_graph()
