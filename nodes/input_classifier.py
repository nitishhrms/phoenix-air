"""
Input Classifier — first node in every pipeline turn.

Uses Claude Haiku to classify each user message as:
  answer   — user is responding to the bot's last question
              (city name, date, passenger name, 'yes', 'no', a number, email/phone)
  question — user is asking for information
              (policy, general travel knowledge, 'what is...', 'how much...')
  query    — user is making a request or starting a new action
              (book a flight, transfer to agent, start over, lodge a complaint)

This classification drives the entire downstream routing:
  answer   → intent_router returns "normal" immediately (route by conv_state)
  question → intent_router runs the policy/general/OOD classification pipeline
  query    → intent_router runs the full intent detection pipeline
"""

import os
from langchain_core.messages import HumanMessage

_BOOKING_STATE_HINTS = {
    "COLLECTING_DEPARTURE":   "The bot just asked: what city are you departing from?",
    "COLLECTING_DESTINATION": "The bot just asked: what city are you flying to?",
    "COLLECTING_DATE":        "The bot just asked: what date would you like to travel?",
    "PRESENTING_FLIGHTS":     "The bot just showed available flights and asked which one to select.",
    "COLLECTING_PASSENGER":   "The bot just asked for the passenger's full name.",
    "COLLECTING_CONTACT":     "The bot just asked for an email address or phone number.",
    "CONFIRMING_BOOKING":     "The bot just asked the user to confirm or cancel the booking.",
    "COLLECTING_PAYMENT":     "The bot just asked for payment details.",
    "POST_BOOKING":           "The booking is complete. The bot is in post-booking state.",
}


_SYSTEM_PROMPT = (
    "You are a classifier for an airline booking chatbot. "
    "Classify the user message into exactly one of: answer, question, query.\n\n"
    "answer   = user is directly responding to what the bot asked "
    "(providing a city, date, name, confirmation number, 'yes', 'no', or a selection like '1')\n"
    "question = user is asking for information "
    "('what is the refund policy?', 'how much does baggage cost?', 'is there wifi?')\n"
    "query    = user is making a request or starting a new action "
    "('book a flight to London', 'transfer me to an agent', 'start over', 'I have a complaint')\n\n"
    "Reply with ONE word only: answer, question, or query."
)


def _classify_with_groq(user_input: str, context: str) -> str:
    """
    Primary classifier: Groq (Llama 3.1 8B) — ~100ms latency, free tier.
    Set GROQ_API_KEY in .env to enable.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        result = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT + f"\n\n{context}"},
                {"role": "user",   "content": user_input},
            ],
            max_tokens=5,
            temperature=0,
        ).choices[0].message.content.strip().lower().split()[0]
        return result if result in ("answer", "question", "query") else ""
    except Exception as e:
        print(f"[CLASSIFIER] Groq error: {e}")
        return ""


def _classify_with_haiku(user_input: str, context: str) -> str:
    """
    Fallback classifier: Claude Haiku — reliable but ~600ms.
    Used when Groq is unavailable or errors.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "query"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        result = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            system=_SYSTEM_PROMPT + f"\n\n{context}",
            messages=[{"role": "user", "content": user_input}],
        ).content[0].text.strip().lower().split()[0]
        return result if result in ("answer", "question", "query") else "query"
    except Exception as e:
        print(f"[CLASSIFIER] Haiku error: {e}")
        return "query"


def _llm_classify(user_input: str, conv_state: str, last_bot_msg: str) -> str:
    """
    Classify input as answer / question / query.
    Tries Groq first (fast), falls back to Haiku (reliable).
    """
    state_hint = _BOOKING_STATE_HINTS.get(conv_state, f"Current state: {conv_state}.")
    context_lines = [f"Bot state: {conv_state}.", state_hint]
    if last_bot_msg:
        context_lines.append(f"Last bot message: {last_bot_msg[:250]}")
    context = "\n".join(context_lines)

    # Primary: Groq (fast small model)
    result = _classify_with_groq(user_input, context)
    if result:
        print(f"[CLASSIFIER] Groq: {result}")
        return result

    # Fallback: Haiku
    result = _classify_with_haiku(user_input, context)
    print(f"[CLASSIFIER] Haiku: {result}")
    return result


# Phrases that are always a query regardless of booking state or context.
# Checked before the LLM to avoid misclassification when the bot is expecting an answer.
_QUERY_SHORTCUTS = [
    # field correction requests
    "update my", "i want to update", "i need to update",
    "want to change my", "need to change my", "i want to change my",
    "i want to correct", "i need to correct",
    # booking actions
    "start over", "restart", "begin again", "book another",
    "speak to a human", "talk to an agent", "live agent", "real person",
    # greetings
    "hello", "hi there", "hey there",
]


def input_classifier_node(state: dict) -> dict:
    user_input = state.get("user_input", "").strip()
    conv_state = state.get("conv_state", "IDLE")

    # Add user message to conversation history (moved here from intent_router)
    new_messages = state.get("messages", []) + [HumanMessage(content=user_input)]

    # During auth states input is always an answer (language choice, phone, OTP)
    if conv_state in ("SELECTING_LANGUAGE", "COLLECTING_PHONE", "VERIFYING_OTP"):
        print(f"[CLASSIFIER] auth state -> answer")
        return {**state, "input_type": "answer", "messages": new_messages}

    # Single-digit during flight selection is obviously an answer
    if user_input in ("1", "2", "3", "4", "5") and conv_state == "PRESENTING_FLIGHTS":
        print(f"[CLASSIFIER] flight selection -> answer")
        return {**state, "input_type": "answer", "messages": new_messages}

    # Pre-LLM shortcut: unambiguous query phrases — always a request, never an answer
    t = user_input.lower()
    if any(kw in t for kw in _QUERY_SHORTCUTS):
        print(f"[CLASSIFIER] shortcut -> query")
        return {**state, "input_type": "query", "messages": new_messages}

    # Get most recent bot message to give LLM context about what was asked
    last_bot = ""
    for msg in reversed(state.get("messages", [])):
        if msg.__class__.__name__ == "AIMessage":
            last_bot = msg.content[:250]
            break

    input_type = _llm_classify(user_input, conv_state, last_bot)
    print(f"[CLASSIFIER] '{user_input[:60]}' -> {input_type}")

    return {**state, "input_type": input_type, "messages": new_messages}
