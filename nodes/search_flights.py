"""
Parses the user's date(s), validates, calls the external API,
runs Chain-of-Thought flight analysis, and stores ranked results.

Accepts dates in virtually any format:
  MM/DD/YYYY, DD-MM-YYYY, "August 15", "15th August 2026",
  "tomorrow", "next Friday", YYYY-MM-DD, etc.

CoT analysis: after fetching flights, Claude Haiku thinks step-by-step
to label each flight as best_value / cheapest / fastest.
"""

import os
import re
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from langchain_core.messages import AIMessage
from services.external_api import search_flights, search_flights_multi
from services.llm import chat_response, cot_analyze_flights

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def _from_iso(m) -> str | None:
    try:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except Exception:
        return None


def _from_month_day_year(m) -> str | None:
    """Handles 'August 15 2026' and 'August 15th 2026'."""
    try:
        mon_name = m.group(1).lower()
        day      = int(m.group(2))
        year     = int(m.group(3))
        mon      = _MONTH_MAP.get(mon_name[:3]) or _MONTH_MAP.get(mon_name)
        if not mon:
            return None
        return datetime(year, mon, day).strftime("%Y-%m-%d")
    except Exception:
        return None


def _from_mdy(m) -> str | None:
    try:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except Exception:
        return None


def _from_ordinal(m) -> str | None:
    try:
        day      = int(m.group(1))
        mon      = _MONTH_MAP.get(m.group(2).lower())
        year_str = m.group(3)
        if not mon:
            return None
        if year_str:
            year = int(year_str)
        else:
            today = datetime.utcnow().date()
            year = today.year
            candidate = datetime(year, mon, day).date()
            if candidate < today:
                year += 1
        return datetime(year, mon, day).strftime("%Y-%m-%d")
    except Exception:
        return None


# Pre-compiled regex patterns (tried in order — first match wins)
_DATE_PATTERNS = [
    # YYYY-MM-DD (ISO): "2026-08-15"
    (re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'), lambda m: _from_iso(m)),
    # Month Day Year: "August 15 2026", "August 15th 2026"
    (re.compile(r'\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})\b', re.IGNORECASE), lambda m: _from_month_day_year(m)),
    # Day-Month-Year numeric: "15/08/2026", "08-15-2026"
    (re.compile(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b'), lambda m: _from_mdy(m)),
    # "15th August 2026"
    (re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)\s+([A-Za-z]+)(?:\s+(\d{4}))?\b'), lambda m: _from_ordinal(m)),
]

_DATE_FORMAT_REMINDER = (
    "Please enter the date as: Month Day Year — for example, August 15 2026."
)


_DATE_CONTENT_RE = re.compile(
    r'\d|'
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
    r'|january|february|march|april|june|july|august|september|october|november|december'
    r'|today|tomorrow|yesterday|next|this'
    r'|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    re.IGNORECASE,
)


def _has_date_content(text: str) -> bool:
    return bool(_DATE_CONTENT_RE.search(text))


def _parse_date(text: str) -> str | None:
    """Return YYYY-MM-DD or None. Tries regex → dateutil → LLM."""
    for pattern, handler in _DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            result = handler(m)
            if result:
                return result

    if _has_date_content(text):
        try:
            dt = dateparser.parse(text, fuzzy=True)
            if dt:
                today = datetime.utcnow().date()
                if dt.date() < today and str(datetime.utcnow().year) not in text and str(datetime.utcnow().year + 1) not in text:
                    dt = dt.replace(year=dt.year + 1)
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            today = datetime.utcnow().strftime("%Y-%m-%d")
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=16,
                system=(
                    f"Today is {today}. Extract the travel date from the user message. "
                    "Reply ONLY with YYYY-MM-DD or UNKNOWN."
                ),
                messages=[{"role": "user", "content": text}],
            )
            result = msg.content[0].text.strip()
            if result != "UNKNOWN" and re.fullmatch(r'\d{4}-\d{2}-\d{2}', result):
                return result
        except Exception:
            pass

    return None


def _validate_date(date_str: str) -> str | None:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.utcnow().date()
        if dt < today:
            return "That date is in the past."
        if dt > today + timedelta(days=365):
            return "I can only book flights up to one year in advance."
        return None
    except ValueError:
        return "I didn't understand that date."


import heapq


def _parse_multiple_dates(text: str) -> list[str]:
    text_lower = text.lower()

    range_match = re.search(
        r'(\w+\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}(?:st|nd|rd|th)?\s+\w+)'
        r'\s+(?:to|through|until|–|-)\s+'
        r'(\w+\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}(?:st|nd|rd|th)?\s+\w+)',
        text_lower
    )
    if range_match:
        start = _parse_date(range_match.group(1))
        end   = _parse_date(range_match.group(2))
        if start and end and not _validate_date(start) and not _validate_date(end):
            s = datetime.strptime(start, "%Y-%m-%d")
            e = datetime.strptime(end,   "%Y-%m-%d")
            if s <= e:
                delta = (e - s).days + 1
                if 1 < delta <= 3:
                    return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta)]

    or_parts = re.split(r'\s+or\s+', text_lower)
    if len(or_parts) >= 2:
        dates = [_parse_date(p.strip()) for p in or_parts]
        dates = [d for d in dates if d and not _validate_date(d)]
        if len(dates) >= 2:
            return dates[:3]

    return []


def _rank_flights(flights: list) -> list:
    """Score each flight by value = price + duration_minutes * 0.5. Sort best-value first."""
    if not flights:
        return flights
    scored = [(float(f.get("price", 9999)) + float(f.get("durationMinutes", 0)) * 0.5, f)
              for f in flights]
    scored.sort(key=lambda x: x[0])
    result = []
    for i, (_, f) in enumerate(scored):
        flight = dict(f)
        flight["recommended"] = (i == 0)
        result.append(flight)
    return result


def _apply_cot_tags(flights: list, cot: dict) -> list:
    """Apply best_value / cheapest / fastest flags from CoT analysis."""
    for i, f in enumerate(flights):
        f["best_value"] = (i == cot.get("best_value_idx", 0))
        f["cheapest"]   = (i == cot.get("cheapest_idx",   0))
        f["fastest"]    = (i == cot.get("fastest_idx",    0))
    return flights


def search_flights_node(state: dict) -> dict:
    user_input  = state.get("user_input", "")
    messages    = state.get("messages", [])
    retry_count = state.get("retry_count", 0)
    src         = state.get("departure_iata", "")
    dst         = state.get("destination_iata", "")
    src_city    = state.get("departure_city", src)
    dst_city    = state.get("destination_city", dst)
    language    = state.get("language", "en")

    # ── Parse multiple dates first ─────────────────────────────────
    multi_dates = _parse_multiple_dates(user_input)

    if multi_dates:
        multi_result = search_flights_multi(src, dst, multi_dates)
        all_flights = []
        for date, date_flights in multi_result.get("flights_by_date", {}).items():
            for f in date_flights:
                f["travelDate"] = date
            all_flights.extend(date_flights)

        if not all_flights:
            ctx      = {"departure": src_city, "destination": dst_city, "dates": ", ".join(multi_dates)}
            task     = "No flights found for those dates. Apologise and let them know."
            fallback = f"Sorry, no flights found from {src_city} to {dst_city} on those dates."
            response = chat_response(task, ctx, user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {**state, "travel_dates": multi_dates, "conv_state": "DONE",
                    "end_call": True, "response": response, "messages": messages + [msg]}

        ranked = _rank_flights(all_flights)

        # ── CoT analysis ───────────────────────────────────────────
        cot = cot_analyze_flights(ranked, src_city, dst_city, ", ".join(multi_dates))
        if cot:
            ranked = _apply_cot_tags(ranked, cot)

        ctx      = {"flight_count": len(ranked), "date_count": len(multi_dates), "dates": ", ".join(multi_dates)}
        task     = f"Found {len(ranked)} flights across {len(multi_dates)} dates. Tell the user and ask them to select one from the cards shown."
        fallback = f"I found {len(ranked)} flight(s) across {len(multi_dates)} date(s). Please select a flight from the options shown."
        if cot.get("summary"):
            task += f" CoT insight: {cot['summary']}"
        response = chat_response(task, ctx, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {
            **state,
            "conv_state":   "PRESENTING_FLIGHTS",
            "travel_date":  multi_dates[0],
            "travel_dates": multi_dates,
            "flights":      ranked,
            "cot_analysis": cot,
            "retry_count":  0,
            "response":     response,
            "messages":     messages + [msg],
        }

    # ── Single date path ───────────────────────────────────────────
    date_str = _parse_date(user_input)
    if not date_str:
        retry_count += 1
        if retry_count >= 3:
            task     = "User couldn't provide a valid date after 3 tries. Apologise and end."
            fallback = "I'm having trouble understanding the date. Please call back."
            response = chat_response(task, {}, user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {**state, "response": response, "end_call": True, "messages": messages + [msg]}

        if retry_count == 1:
            # First failure: hardcoded format reminder — fast and predictable
            print("[HARDCODED] search_flights: date format reminder")
            msg = AIMessage(content=_DATE_FORMAT_REMINDER)
            return {
                **state,
                "response":      _DATE_FORMAT_REMINDER,
                "response_type": "hardcoded",
                "retry_count":   retry_count,
                "messages":      messages + [msg],
            }

        # Second failure: LLM responds gently
        task     = "Couldn't parse a date again. Gently ask them to say it like: August 15 2026."
        fallback = "Still having trouble with that date — could you say it like: August 15 2026?"
        response = chat_response(task, {}, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {**state, "response": response, "retry_count": retry_count, "messages": messages + [msg]}

    date_error = _validate_date(date_str)
    if date_error:
        retry_count += 1
        ctx      = {"date_entered": date_str, "validation_error": date_error}
        task     = f"Date validation failed: {date_error}. Tell the user and ask for a date within the next year."
        fallback = f"{date_error} Please provide a date between today and one year from now."
        response = chat_response(task, ctx, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {**state, "response": response, "retry_count": retry_count, "messages": messages + [msg]}

    result = search_flights(src, dst, date_str)

    if "error" in result:
        code = result.get("code", "")
        if code == "NO_FLIGHTS":
            ctx      = {"departure": src_city, "destination": dst_city, "date": date_str}
            task     = f"No flights from {src_city} to {dst_city} on {date_str}. Apologise."
            fallback = f"I'm sorry, no flights available from {src_city} to {dst_city} on {date_str}."
            response = chat_response(task, ctx, user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {**state, "travel_date": date_str, "conv_state": "DONE",
                    "end_call": True, "response": response, "messages": messages + [msg]}
        if code == "INVALID_DATE":
            task     = "Date rejected by flight system. Ask for a valid future date."
            fallback = "That date isn't valid. Please provide a future travel date."
            response = chat_response(task, {}, user_input, language=language) or fallback
            msg = AIMessage(content=response)
            return {**state, "response": response, "retry_count": retry_count + 1, "messages": messages + [msg]}

        task     = "Flight system temporarily unavailable. Ask them to try again shortly."
        fallback = "I'm having trouble reaching our flight system. Please try again shortly."
        response = chat_response(task, {}, user_input, language=language) or fallback
        msg = AIMessage(content=response)
        return {**state, "response": response, "end_call": True, "messages": messages + [msg]}

    flights = result.get("flights", [])
    ranked  = _rank_flights(flights)

    # ── CoT analysis ───────────────────────────────────────────────
    cot = cot_analyze_flights(ranked, src_city, dst_city, date_str)
    if cot:
        ranked = _apply_cot_tags(ranked, cot)

    ctx  = {"departure": src_city, "destination": dst_city, "travel_date": date_str, "flight_count": len(ranked)}
    task = f"Found {len(ranked)} flights from {src_city} to {dst_city} on {date_str}. Tell the user and ask them to pick one from the cards."
    if cot.get("summary"):
        task += f" CoT insight: {cot['summary']}"
    fallback = f"I found {len(ranked)} flight(s) from {src_city} to {dst_city} on {date_str}. Please select one from the options shown."
    response = chat_response(task, ctx, user_input, language=language) or fallback
    msg = AIMessage(content=response)
    return {
        **state,
        "conv_state":   "PRESENTING_FLIGHTS",
        "travel_date":  date_str,
        "flights":      ranked,
        "cot_analysis": cot,
        "retry_count":  0,
        "response":     response,
        "messages":     messages + [msg],
    }


def _fmt_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return iso
