"""
Authentication node — handles COLLECTING_PHONE and VERIFYING_OTP states.

Phone stage rules:
  - Rule-based phone validation (regex) → LLM fallback for natural phrasing
  - Off-topic inputs (policy/general/OOD questions) → hardcoded redirect, no LLM
  - Supports international formats: +1-408-555-1234, (408) 555.1234, +44 20 7946 0958

OTP stage rules:
  - Input must look like digits — anything else gets a hardcoded redirect
  - Wrong code → LLM response with attempt count
  - Too many failures → end session
"""

import re
from langchain_core.messages import AIMessage
from services.auth import generate_otp, verify_otp, send_otp
from services.llm import chat_response

MAX_OTP_ATTEMPTS = 3

# ── Hardcoded redirects (no LLM — fast, predictable, secure) ──────────────────

_PHONE_REDIRECT = (
    "I need your phone number to continue. "
    "Please enter it in international format, e.g. +1 408 555 1234."
)

_OTP_REDIRECT = (
    "Please enter the 6-digit verification code we sent to your phone. "
    "It should be a number like 123456."
)

# Signals that user typed a question instead of a phone number
_QUESTION_SIGNALS = [
    "?", "what", "how", "why", "when", "where", "who",
    "can you", "do you", "is there", "tell me", "explain",
    "refund", "cancel", "policy", "baggage", "wifi", "flight",
    "help", "support",
]


def _clean_phone(text: str) -> str | None:
    """Strip formatting and validate as international phone number."""
    digits = re.sub(r'[\s\-().+]+', '', text)
    # Keep leading + if present in original
    if text.strip().startswith('+'):
        digits = '+' + digits
    # Also handle dots: 408.555.1234
    digits = re.sub(r'\.', '', digits)
    # Re-strip after dot removal
    digits = re.sub(r'[\s\-()]+', '', digits)
    if re.fullmatch(r'\+?[0-9]{7,15}', digits):
        return digits
    return None


def _looks_like_question(text: str) -> bool:
    """Return True if input looks like a question rather than a phone number."""
    t = text.lower().strip()
    return any(sig in t for sig in _QUESTION_SIGNALS)


def _extract_digits(text: str) -> str:
    """Extract only digit characters from text."""
    return re.sub(r'\D', '', text)


def auth_node(state: dict) -> dict:
    conv_state  = state.get("conv_state", "COLLECTING_PHONE")
    user_input  = state.get("user_input", "").strip()
    messages    = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    language    = state.get("language", "en")

    # ── Step 1: collect phone ──────────────────────────────────────────────────
    if conv_state == "COLLECTING_PHONE":

        # Off-topic input → hardcoded redirect (no LLM, no "invalid phone" confusion)
        if user_input and _looks_like_question(user_input):
            print("[HARDCODED] auth: off-topic redirect at phone stage")
            msg = AIMessage(content=_PHONE_REDIRECT)
            return {
                **state,
                "response":      _PHONE_REDIRECT,
                "response_type": "hardcoded",
                "messages":      messages + [msg],
            }

        phone = _clean_phone(user_input) if user_input else None

        if not phone:
            # Rule-based response first; LLM makes it natural
            if not user_input:
                task     = "Ask the user to provide their phone number to begin booking. Mention international format e.g. +1 408 555 1234."
                fallback = "To get started, please provide your phone number in international format, e.g. +1 408 555 1234."
            else:
                task     = "The input was not a valid phone number. Ask them to enter it again in international format e.g. +1 408 555 1234. Accept formats from any country."
                fallback = "That doesn't look like a valid phone number. Please try again in international format, e.g. +1 408 555 1234."
            response = chat_response(task, {}, user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {**state, "response": response, "messages": messages + [msg]}

        # Valid phone — generate + send OTP
        code       = generate_otp(phone)
        otp_result = send_otp(phone, code)

        if otp_result == "sms":
            task     = f"A 6-digit SMS verification code was sent to {phone}. Tell the user to check their messages and enter the code."
            fallback = f"I've sent a 6-digit code to {phone}. Please check your messages and enter it to continue."
        elif otp_result == "email":
            task     = "A 6-digit verification code was sent to the user's email address. Tell the user to check their inbox (and spam folder) and enter the code."
            fallback = "I've sent a 6-digit verification code to your email. Please check your inbox and enter it here."
        else:
            task     = f"We were unable to send a code externally. Give the user their verification code: {code}, and ask them to enter it now to continue."
            fallback = f"Your verification code is: {code}. Please enter it to continue."

        response = chat_response(task, {}, user_input, max_tokens=80, language=language) or fallback

        if otp_result is False and str(code) not in response:
            response = f"{response}\n{fallback}" if response != fallback else fallback

        msg = AIMessage(content=response)
        return {
            **state,
            "conv_state":  "VERIFYING_OTP",
            "auth_phone":  phone,
            "retry_count": 0,
            "response":    response,
            "messages":    messages + [msg],
        }

    # ── Step 2: verify OTP ────────────────────────────────────────────────────
    if conv_state == "VERIFYING_OTP":
        auth_phone = state.get("auth_phone", "")
        only_digits = _extract_digits(user_input)

        # Input has no digits or is too short/long → hardcoded redirect
        if len(only_digits) != 6:
            print("[HARDCODED] auth: non-OTP input at OTP stage")
            msg = AIMessage(content=_OTP_REDIRECT)
            return {
                **state,
                "response":      _OTP_REDIRECT,
                "response_type": "hardcoded",
                "messages":      messages + [msg],
            }

        if verify_otp(auth_phone, only_digits):
            task     = "Phone number verified successfully. Welcome the user warmly and ask which city they are departing from."
            fallback = "Verified! Welcome to Phoenix Air. Which city are you departing from?"
            response = chat_response(task, {}, user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {
                **state,
                "authenticated": True,
                "caller_phone":  auth_phone,
                "conv_state":    "COLLECTING_DEPARTURE",
                "retry_count":   0,
                "response":      response,
                "messages":      messages + [msg],
            }

        retry_count += 1
        if retry_count >= MAX_OTP_ATTEMPTS:
            task     = "The user failed OTP verification too many times. Politely end the session."
            fallback = "Too many incorrect attempts. Please try again later. Goodbye!"
            response = chat_response(task, {}, user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {
                **state,
                "end_call":    True,
                "retry_count": retry_count,
                "response":    response,
                "messages":    messages + [msg],
            }

        remaining = MAX_OTP_ATTEMPTS - retry_count
        task     = f"The OTP code was wrong. Tell the user and let them know they have {remaining} attempt(s) left."
        fallback = f"That code doesn't match. Please try again ({remaining} attempt(s) left)."
        response = chat_response(task, {}, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {
            **state,
            "retry_count": retry_count,
            "response":    response,
            "messages":    messages + [msg],
        }

    # Fallback — shouldn't reach here
    fallback = "Please provide your phone number to get started."
    response = chat_response("Ask the user for their phone number.", {}, user_input, language=language) or fallback
    msg = AIMessage(content=response)
    return {**state, "conv_state": "COLLECTING_PHONE", "response": response, "messages": messages + [msg]}
