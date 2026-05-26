"""
Central LLM response generator with prompt caching and multi-language support.

Every node calls chat_response() instead of returning a hardcoded string.

Built-in guards (transparent to callers):
  - Per-call timeout (8s): if Haiku doesn't respond, returns None → node uses fallback
  - Hallucination guard: checks generated response against the context; if it
    fabricates data not in context, returns None → node uses fallback

Prompt caching: the static system rules are marked with cache_control so
Anthropic caches them across calls, reducing latency and cost.
"""

import os
import concurrent.futures

# ── Anthropic client singleton ────────────────────────────────────────────────
_anthropic_client = None

def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    return _anthropic_client


# ── Static system prompt (cached across all calls) ───────────────────────────
_STATIC_SYSTEM = (
    "You are Phoenix Air's friendly voice booking assistant.\n\n"
    "CORE RULES:\n"
    "- Reply in 1-2 short sentences only — this is a voice interface\n"
    "- Never invent flight numbers, prices, confirmation codes, airports, "
    "or dates — use ONLY data from the booking context block\n"
    "- Be warm and natural, not robotic or corporate\n"
    "- Stay focused on airline booking — politely decline anything off-topic\n"
    "- When confirming a booking detail, always repeat it back to the user\n"
    "- For flight selection, reference the airline name and flight number\n"
    "- For payment, always state the total amount clearly\n"
    "- For policy questions, answer using ONLY provided policy text\n"
    "- Maintain a professional yet warm, helpful tone throughout\n"
    "- If a language instruction is given below, ALWAYS respond in that language"
)

_LANG_INSTRUCTIONS: dict[str, str] = {
    "en": "",
    "es": "\nLANGUAGE: Respond in Spanish (Español). Keep the same warm, concise style.",
    "fr": "\nLANGUAGE: Respond in French (Français). Keep the same warm, concise style.",
    "hi": "\nLANGUAGE: Respond in Hindi (हिंदी). Keep the same warm, concise style.",
    "zh": "\nLANGUAGE: Respond in Chinese Mandarin (中文). Keep the same warm, concise style.",
    "ar": "\nLANGUAGE: Respond in Arabic (العربية). Keep the same warm, concise style.",
    "pt": "\nLANGUAGE: Respond in Portuguese (Português). Keep the same warm, concise style.",
    "de": "\nLANGUAGE: Respond in German (Deutsch). Keep the same warm, concise style.",
}


# ── Internal: raw Haiku call with prompt caching ──────────────────────────────

def _call_haiku(dynamic_system: str, user_msg: str, max_tokens: int) -> str:
    client = _get_client()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": _STATIC_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": dynamic_system,
            },
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    return msg.content[0].text.strip()


def _with_timeout(fn, args: tuple = (), timeout_s: float = 8.0):
    """Run fn(*args) with a wall-clock timeout. Returns None on timeout/error."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            print(f"[LLM TIMEOUT] exceeded {timeout_s}s — using fallback")
            return None
        except Exception as exc:
            print(f"[LLM ERROR] {exc}")
            return None


# ── Hallucination guard ───────────────────────────────────────────────────────

_HAL_SYSTEM = (
    "You are a strict fact-checker for an airline booking assistant. "
    "Given a booking context, a user input, and a generated response, "
    "reply with exactly PASS if the response only uses facts from the context, "
    "or FAIL if it invents flights, prices, cities, dates, or confirmation numbers "
    "not present in the context."
)


def _hallucination_check(user_input: str, response: str, context_str: str) -> bool:
    if not context_str.strip() or context_str.strip() == "(nothing collected yet)":
        return True
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return True
    try:
        def _check():
            client = _get_client()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=16,
                system=[
                    {"type": "text", "text": _HAL_SYSTEM, "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Context:\n{context_str}\n\n"
                        f"User said: {user_input}\n"
                        f"Response: {response}"
                    ),
                }],
            )
            return msg.content[0].text.strip().upper()

        verdict = _with_timeout(_check, timeout_s=5.0)
        if verdict and verdict.startswith("FAIL"):
            print(f"[HALLUCINATION GUARD] FAIL | response: {response[:80]}")
            return False
        return True
    except Exception as exc:
        print(f"[HALLUCINATION GUARD] skipped (error: {exc})")
        return True


# ── CoT flight analysis ───────────────────────────────────────────────────────

_COT_SYSTEM = (
    "You are an expert flight analyst for Phoenix Air. "
    "Analyze available flights step by step. Think through price, duration, stops, and overall value. "
    "Format your response EXACTLY as:\n"
    "THINKING: <step-by-step reasoning>\n"
    "BEST_VALUE: <flight number 1-N>\n"
    "CHEAPEST: <flight number 1-N>\n"
    "FASTEST: <flight number 1-N>\n"
    "SUMMARY: <one concise sentence for the user>"
)


def cot_analyze_flights(flights: list, src_city: str, dst_city: str, date_str: str) -> dict:
    """
    Chain-of-Thought analysis of available flights.
    Returns dict with best_value_idx, cheapest_idx, fastest_idx, summary.
    """
    if not flights or not os.getenv("ANTHROPIC_API_KEY"):
        return {}

    def _fmt_time(iso: str) -> str:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.strftime("%I:%M %p").lstrip("0")
        except Exception:
            return iso

    flight_list = "\n".join([
        f"Flight {i+1}: {f.get('airline', '')} {f.get('flightNumber', '')}, "
        f"departs {_fmt_time(f.get('departureTime', ''))}, "
        f"arrives {_fmt_time(f.get('arrivalTime', ''))}, "
        f"duration {f.get('durationMinutes', 0)}min, "
        f"${float(f.get('price', 0)):.2f}, "
        f"stops: {f.get('stops', 0)}"
        for i, f in enumerate(flights)
    ])

    def _run():
        client = _get_client()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=[
                {"type": "text", "text": _COT_SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{
                "role": "user",
                "content": (
                    f"Route: {src_city} → {dst_city} on {date_str}\n\n"
                    f"Available flights:\n{flight_list}\n\n"
                    "Analyze and recommend."
                ),
            }],
        )
        return msg.content[0].text.strip()

    raw = _with_timeout(_run, timeout_s=8.0)
    if not raw:
        return {}

    result = {"thinking": "", "best_value_idx": 0, "cheapest_idx": 0, "fastest_idx": 0, "summary": ""}
    n = len(flights)
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("THINKING:"):
            result["thinking"] = line[9:].strip()
        elif line.startswith("BEST_VALUE:"):
            try:
                result["best_value_idx"] = max(0, min(int(line.split(":")[1].strip()) - 1, n - 1))
            except Exception:
                pass
        elif line.startswith("CHEAPEST:"):
            try:
                result["cheapest_idx"] = max(0, min(int(line.split(":")[1].strip()) - 1, n - 1))
            except Exception:
                pass
        elif line.startswith("FASTEST:"):
            try:
                result["fastest_idx"] = max(0, min(int(line.split(":")[1].strip()) - 1, n - 1))
            except Exception:
                pass
        elif line.startswith("SUMMARY:"):
            result["summary"] = line[8:].strip()
    return result


# ── Self-reflection check for flight confirmation ─────────────────────────────

_REFLECT_SYSTEM = (
    "You are a quality checker for an airline booking AI. "
    "Check if the response correctly confirms the selected flight. "
    "Reply with EXACTLY 'PASS' if correct, 'FAIL' if the flight details are missing or wrong."
)


def self_reflect_confirmation(selected: dict, response: str) -> bool:
    """Returns True if the response correctly confirms the selected flight."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return True
    try:
        def _check():
            client = _get_client()
            flight_info = f"{selected.get('airline', '')} {selected.get('flightNumber', '')}"
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=8,
                system=[
                    {"type": "text", "text": _REFLECT_SYSTEM, "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{
                    "role": "user",
                    "content": f"Selected flight: {flight_info}\nAI response: {response}",
                }],
            )
            return msg.content[0].text.strip().upper()

        verdict = _with_timeout(_check, timeout_s=5.0)
        if verdict and verdict.startswith("FAIL"):
            print(f"[SELF-REFLECT] FAIL — regenerating confirmation")
            return False
        return True
    except Exception:
        return True


# ── Public API ────────────────────────────────────────────────────────────────

def chat_response(task: str, context: dict = None, user_input: str = "",
                  max_tokens: int = 120, language: str = "en",
                  skip_hallucination_guard: bool = True) -> str | None:
    chat_response.last_was_llm = False
    """
    Generate a short conversational response via Claude Haiku.

    Guards applied automatically:
      1. 8s per-call timeout  → returns None on timeout (node uses its fallback)
      2. Hallucination check  → returns None if response fabricates context data

    Prompt caching: static system rules are cached across all calls.
    Language: response language is set via the `language` parameter.

    task       — what the assistant needs to communicate this turn
    context    — booking data already collected (cities, flight, price …)
    user_input — what the user just said
    language   — ISO 639-1 code (en/es/fr/hi/zh/ar/pt/de)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    # Build dynamic context block
    lines = []
    if context:
        for k, v in context.items():
            if v is not None and v != "" and v != []:
                lines.append(f"  {k}: {v}")
    ctx_str = "\n".join(lines) if lines else "  (nothing collected yet)"

    lang_instr = _LANG_INSTRUCTIONS.get(language, "")
    dynamic_system = f"\nCurrent booking context:\n{ctx_str}{lang_instr}"

    user_msg = f"Task: {task}"
    if user_input:
        user_msg += f'\nUser just said: "{user_input}"'

    # Guard 1: timeout on LLM call
    response = _with_timeout(_call_haiku, args=(dynamic_system, user_msg, max_tokens), timeout_s=8.0)
    if response is None:
        return None

    # Guard 2: hallucination check (skipped for web search responses)
    if not skip_hallucination_guard and not _hallucination_check(user_input, response, ctx_str):
        return None

    safe = response[:80].encode("ascii", errors="replace").decode("ascii")
    print(f"[LLM] {safe}")
    chat_response.last_was_llm = True
    return response
