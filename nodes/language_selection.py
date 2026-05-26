"""
Language selection node — first step in every new session.

Presents a language menu, detects the user's choice via keyword matching
with LLM fallback, then advances to COLLECTING_PHONE.
"""

import os
from langchain_core.messages import AIMessage

LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "hi": "हिंदी",
    "zh": "中文",
    "ar": "العربية",
    "pt": "Português",
    "de": "Deutsch",
}

_WELCOME_MSG = (
    "Welcome to Phoenix Air!\n"
    "Please select your language / Seleccione su idioma / Choisissez votre langue:\n"
    "1. English  2. Español  3. Français  4. हिंदी  "
    "5. 中文  6. العربية  7. Português  8. Deutsch"
)

_RETRY_MSG = (
    "I didn't catch that. Please say your language or type a number:\n"
    "1. English  2. Español  3. Français  4. हिंदी  "
    "5. 中文  6. العربية  7. Português  8. Deutsch"
)

# Confirmed messages in each language (shown after selection)
_CONFIRMED = {
    "en": "Great! I'll assist you in English. Please provide your phone number to get started.",
    "es": "¡Perfecto! Te ayudaré en español. Por favor proporciona tu número de teléfono.",
    "fr": "Parfait ! Je vous aiderai en français. Veuillez fournir votre numéro de téléphone.",
    "hi": "बहुत अच्छा! मैं हिंदी में मदद करूंगा। कृपया अपना फ़ोन नंबर दें।",
    "zh": "很好！我将用中文为您服务。请提供您的电话号码。",
    "ar": "رائع! سأساعدك باللغة العربية. يرجى تقديم رقم هاتفك.",
    "pt": "Ótimo! Vou ajudá-lo em português. Por favor, forneça seu número de telefone.",
    "de": "Ausgezeichnet! Ich helfe Ihnen auf Deutsch. Bitte geben Sie Ihre Telefonnummer an.",
}

_DETECT_MAP = {
    "en": ["english", "1", "eng", "anglais", "inglés", "inglés"],
    "es": ["spanish", "español", "espanol", "2", "esp", "castellano"],
    "fr": ["french", "français", "francais", "3", "fre", "fran"],
    "hi": ["hindi", "हिंदी", "4", "hin"],
    "zh": ["chinese", "中文", "mandarin", "5", "chi", "zh", "zhongwen"],
    "ar": ["arabic", "العربية", "6", "ara", "arab"],
    "pt": ["portuguese", "português", "portugues", "7", "por"],
    "de": ["german", "deutsch", "8", "ger", "deu"],
}


def _detect_language(text: str) -> str | None:
    t = text.lower().strip()
    for code, keywords in _DETECT_MAP.items():
        if any(kw in t for kw in keywords):
            return code
    return None


def _llm_detect_language(text: str) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        valid = ", ".join(LANGUAGES.keys())
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4,
            system=(
                f"Detect the language the user wants. "
                f"Valid options: {valid}. Reply with ONLY the 2-letter code."
            ),
            messages=[{"role": "user", "content": text}],
        )
        result = msg.content[0].text.strip().lower()
        return result if result in LANGUAGES else None
    except Exception:
        return None


def language_selection_node(state: dict) -> dict:
    user_input = state.get("user_input", "").strip()
    messages   = state.get("messages", [])

    # First turn: present menu
    if not user_input:
        msg = AIMessage(content=_WELCOME_MSG)
        return {
            **state,
            "conv_state":    "SELECTING_LANGUAGE",
            "response":      _WELCOME_MSG,
            "response_type": "hardcoded",
            "messages":      messages + [msg],
        }

    # Layer 1: keyword detection
    lang = _detect_language(user_input)

    # Layer 2: LLM fallback
    if not lang:
        lang = _llm_detect_language(user_input)

    if not lang:
        print("[HARDCODED] language_selection: retry prompt")
        msg = AIMessage(content=_RETRY_MSG)
        return {**state, "conv_state": "SELECTING_LANGUAGE", "response": _RETRY_MSG, "response_type": "hardcoded", "messages": messages + [msg]}

    response = _CONFIRMED[lang]
    print(f"[HARDCODED] language_selection: confirmed {lang}")
    msg = AIMessage(content=response)
    return {
        **state,
        "language":      lang,
        "conv_state":    "COLLECTING_PHONE",
        "response":      response,
        "response_type": "hardcoded",
        "messages":   messages + [msg],
    }
