"""
Answer Validator — runs when input_type == "answer".

Greetings, transfers, policy questions, and any other non-answer intent can
arrive at ANY booking stage. This node handles that gracefully:

  Step 0: Global intent override
            Checks for greetings, transfers, policy questions, etc.
            If found: updates input_type and routes back to intent_router.
            This ensures queries work at every booking stage even if the
            input_classifier mistakenly labelled them as "answer".

  Step 1: Format validation per booking state (regex L1 → LLM L2)
            L1 Regex  — does the input match the expected format?
            L2 LLM    — spelling correction, natural-language parsing, "did you mean?"

Outcomes:
  validated = True, input_type updated  → intent_router re-routes correctly
  validated = True, user_input updated  → booking node receives corrected input
  validated = False                     → clarification response returned to user

pending_correction stores an LLM-suggested correction across turns:
  Turn 1: "Los Angles" → "Did you mean Los Angeles?"
  Turn 2: "yes"        → correction applied, booking continues
  Turn 2: "no"         → asks again
"""

import re
import os
import json
from langchain_core.messages import AIMessage

_AUTH_STATES = {"COLLECTING_PHONE", "VERIFYING_OTP"}

_UNAMBIGUOUS_QUERY_KWS = [
    "start over", "restart", "begin again", "start again", "reset", "book another",
    "speak to a human", "talk to an agent", "live agent", "real person",
    "hello", "hi there", "hey there", "good morning", "good afternoon", "good evening",
    "update my", "i want to update", "i need to update",
    "want to change my", "need to change my", "i want to change my",
]

_UNAMBIGUOUS_QUESTION_KWS = ["?"]


_DETECT_SYSTEM = (
    "You are a classifier for an airline booking chatbot. "
    "The chatbot currently expects the user to answer a booking question "
    "(e.g. provide a city, date, name, phone number, or flight number).\n\n"
    "Classify the user's message:\n"
    "  question = user is asking for information (policy, baggage, wifi, flights, prices, "
    "refund, cancellation, meals, loyalty, seat, etc.) — even without a question mark\n"
    "  query    = user is making a request (greeting, transfer to agent, restart, complaint, "
    "start over, change something)\n"
    "  answer   = user is directly responding to what the bot asked "
    "(providing a city, date, name, phone number, flight selection, 'yes', 'no', or "
    "saying they will provide the info soon)\n\n"
    "Reply with ONE word only: question, query, or answer."
)


def _llm_detect_intent(user_input: str, conv_state: str) -> tuple[str | None, bool]:
    """
    Use Haiku to detect if the input is actually a question or query rather than a booking answer.
    Returns (input_type, should_reroute) or (None, False).
    """
    t = user_input.lower().strip()
    if any(kw in t for kw in _UNAMBIGUOUS_QUERY_KWS):
        return "query", True
    if any(kw in t for kw in _UNAMBIGUOUS_QUESTION_KWS):
        return "question", True

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, False
    try:
        import anthropic
        context = f"Bot state: {conv_state}."
        raw = anthropic.Anthropic(api_key=api_key).messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            system=_DETECT_SYSTEM + "\n\n" + context,
            messages=[{"role": "user", "content": user_input}],
        ).content[0].text.strip().lower().split()[0]
        if raw in ("question", "query"):
            print(f"[VALIDATOR] Haiku: '{user_input[:40]}' -> {raw}")
            return raw, True
        return None, False
    except Exception as e:
        print(f"[VALIDATOR] Haiku error: {e}")
    return None, False


# ── Regex patterns ────────────────────────────────────────────────────────────

_CITY_RE      = re.compile(r"^[A-Za-z\s,.''\-]+$")
_NAME_RE      = re.compile(r"^[A-Za-z\s''\-\.]+$")
_EMAIL_RE     = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_PHONE_RE     = re.compile(r"^\+?[\d\s\-().]{7,20}$")
_FLIGHT_RE    = re.compile(r"^[1-5]$")
_YES_RE       = re.compile(r"\b(yes|yeah|yep|yup|sure|ok|okay|confirm|correct|right|affirmative)\b", re.I)
_NO_RE        = re.compile(r"\b(no|nope|nah|not|wrong|different|incorrect)\b", re.I)

_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b"),
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}\s*,?\s*\d{4}\b", re.I),
    re.compile(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b", re.I),
    re.compile(r"\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)\b", re.I),
    re.compile(r"\b(tomorrow|today|this\s+(?:friday|saturday|sunday|monday|tuesday|wednesday|thursday))\b", re.I),
]

_STATE_EXPECTED = {
    "COLLECTING_DEPARTURE":   "a departure city name",
    "COLLECTING_DESTINATION": "a destination city name",
    "COLLECTING_DATE":        "a travel date (e.g. August 15 2026 or 2026-08-15)",
    "COLLECTING_PASSENGER":   "a full passenger name with first and last name",
    "COLLECTING_CONTACT":     "an email address or phone number",
    "PRESENTING_FLIGHTS":     "a flight selection number (1, 2, or 3)",
}


# ── L1: Regex validation ──────────────────────────────────────────────────────

def _regex_validate(user_input: str, conv_state: str) -> tuple[bool, str]:
    """Returns (is_valid, user-facing hint when invalid)."""
    text = user_input.strip()

    if conv_state in ("COLLECTING_DEPARTURE", "COLLECTING_DESTINATION"):
        if _CITY_RE.match(text) and len(text.strip()) >= 2:
            return True, ""
        return False, "Please provide a valid city name."

    if conv_state == "COLLECTING_DATE":
        lower = text.lower()
        for pat in _DATE_PATTERNS:
            if pat.search(lower):
                return True, ""
        return False, "Please provide a date like 'August 15 2026' or '2026-08-15'."

    if conv_state == "COLLECTING_PASSENGER":
        if _NAME_RE.match(text) and len(text.split()) >= 2:
            return True, ""
        return False, "Please provide your full name (first and last name)."

    if conv_state == "COLLECTING_CONTACT":
        if _EMAIL_RE.match(text) or _PHONE_RE.match(text):
            return True, ""
        return False, "Please provide a valid email address or phone number (e.g. john@example.com or +1 555 000 0000)."

    if conv_state == "PRESENTING_FLIGHTS":
        if _FLIGHT_RE.match(text):
            return True, ""
        return False, "Please select a flight by entering its number — 1, 2, or 3."

    return True, ""


# ── L2: LLM interpretation and spelling correction ────────────────────────────

_INTERPRET_SYSTEM = (
    "You are validating a user's input for an airline chatbot. "
    "Fix any spelling mistakes and interpret the input. "
    "Return JSON only:\n"
    '{"corrected": "<corrected value or empty string if cannot interpret>", '
    '"confident": <true if certain, false if guessing>}\n\n'
    "Examples:\n"
    '  "Los Angles" -> {"corrected": "Los Angeles", "confident": true}\n'
    '  "Jhn Smth"   -> {"corrected": "John Smith", "confident": false}\n'
    '  "nxt tues"   -> {"corrected": "next tuesday", "confident": true}\n'
    '  "xyz123!!"   -> {"corrected": "", "confident": false}'
)


def _parse_json_result(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start != -1 and end > start:
        parsed = json.loads(raw[start:end])
        return {
            "corrected": str(parsed.get("corrected", "")).strip(),
            "confident": bool(parsed.get("confident", False)),
        }
    return {"corrected": "", "confident": False}


def _llm_interpret(user_input: str, conv_state: str, expected: str) -> dict:
    """Spell-correct / interpret the input using Haiku."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"corrected": "", "confident": False}
    try:
        import anthropic
        system = _INTERPRET_SYSTEM + f"\n\nThe bot currently expects: {expected}."
        raw = anthropic.Anthropic(api_key=api_key).messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            system=system,
            messages=[{"role": "user", "content": user_input}],
        ).content[0].text.strip()
        return _parse_json_result(raw)
    except Exception as e:
        print(f"[VALIDATOR] Haiku interpret error: {e}")
    return {"corrected": "", "confident": False}


def _llm_parse_date(user_input: str) -> str:
    """Convert natural-language date to YYYY-MM-DD using Haiku."""
    from datetime import date
    today  = date.today().isoformat()
    system = (
        f"Today's date is {today}. "
        "Convert the user's date input to YYYY-MM-DD format. "
        "Reply with ONLY the date in YYYY-MM-DD format, or 'invalid' if it cannot be converted."
    )
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""
    try:
        import anthropic
        raw = anthropic.Anthropic(api_key=api_key).messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            system=system,
            messages=[{"role": "user", "content": user_input}],
        ).content[0].text.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
            return raw
    except Exception as e:
        print(f"[VALIDATOR] Haiku date error: {e}")
    return ""


# ── Main node ─────────────────────────────────────────────────────────────────

def answer_validator_node(state: dict) -> dict:
    user_input = state.get("user_input", "").strip()
    conv_state = state.get("conv_state", "IDLE")
    pending    = state.get("pending_correction")
    messages   = state.get("messages", [])

    # ── Pending correction confirmation flow ──────────────────────────────────
    if pending:
        if _YES_RE.search(user_input):
            print(f"[VALIDATOR] correction confirmed: '{user_input}' -> '{pending}'")
            return {
                **state,
                "user_input":         pending,
                "pending_correction": None,
                "validated":          True,
            }
        elif _NO_RE.search(user_input):
            expected = _STATE_EXPECTED.get(conv_state, "your answer")
            resp = f"No problem! Could you please provide {expected} again?"
            msg  = AIMessage(content=resp)
            return {
                **state,
                "pending_correction": None,
                "validated":          False,
                "response":           resp,
                "response_type":      "hardcoded",
                "messages":           messages + [msg],
            }
        else:
            state = {**state, "pending_correction": None}

    # ── Global intent override ────────────────────────────────────────────────
    new_type, reroute = _llm_detect_intent(user_input, conv_state)
    if reroute:
        print(f"[VALIDATOR] rerouting '{user_input[:40]}' as {new_type}")
        return {**state, "input_type": new_type, "validated": True}

    if conv_state not in _STATE_EXPECTED:
        return {**state, "validated": True}

    expected = _STATE_EXPECTED[conv_state]

    # ── L1: Regex validation ──────────────────────────────────────────────────
    is_valid, hint = _regex_validate(user_input, conv_state)

    if is_valid:
        if conv_state == "COLLECTING_DATE":
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", user_input):
                normalised = _llm_parse_date(user_input)
                if normalised:
                    print(f"[VALIDATOR] date normalised: '{user_input}' -> '{normalised}'")
                    return {**state, "user_input": normalised, "validated": True}
        return {**state, "validated": True}

    # ── L2: Regex failed — LLM tries to interpret ─────────────────────────────
    print(f"[VALIDATOR] regex failed at {conv_state}: '{user_input}' — calling LLM")
    correction = _llm_interpret(user_input, conv_state, expected)

    if correction["corrected"]:
        corrected = correction["corrected"]
        if correction["confident"]:
            resp = f"Did you mean '{corrected}'? Say yes to continue or provide the correct information."
            print(f"[VALIDATOR] suggesting: '{corrected}' (confident)")
        else:
            resp = f"I'm not sure — did you mean '{corrected}'? Say yes to continue or type it again."
            print(f"[VALIDATOR] suggesting: '{corrected}' (uncertain)")

        msg = AIMessage(content=resp)
        return {
            **state,
            "validated":          False,
            "pending_correction": corrected,
            "response":           resp,
            "response_type":      "hardcoded",
            "messages":           messages + [msg],
        }

    resp = f"I didn't quite catch that. {hint}"
    msg  = AIMessage(content=resp)
    return {
        **state,
        "validated":          False,
        "pending_correction": None,
        "response":           resp,
        "response_type":      "hardcoded",
        "messages":           messages + [msg],
    }
