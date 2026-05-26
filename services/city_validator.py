"""
LLM-based city existence validator — Layer 4 of airport resolution.

Used only when rule-based and embedding layers both fail to find a match.
Asks Haiku whether the input is a real city/region with a major airport.

Returns:
  (True,  "Official City Name")  — real city with airport
  (False, None)                  — not a recognizable city / no airport
"""

import os


def llm_check_city(city_input: str) -> tuple[bool, str | None]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return False, None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            system=(
                "You are an airport lookup assistant. "
                "Determine if the input is a real city or region that has a major airport. "
                "Reply EXACTLY in one of these two formats:\n"
                "YES <Official City Name>\n"
                "NO\n"
                "Examples:\n"
                "  Sanfransisco → YES San Francisco\n"
                "  Chi-town     → YES Chicago\n"
                "  Big Apple    → YES New York\n"
                "  Xyz123       → NO\n"
                "  randomword   → NO"
            ),
            messages=[{"role": "user", "content": city_input.strip()}],
        )
        raw = msg.content[0].text.strip()
        if raw.upper().startswith("YES"):
            corrected = raw[3:].strip()
            return True, corrected if corrected else city_input
        return False, None
    except Exception as e:
        print(f"[CITY VALIDATOR] error: {e}")
        return False, None
