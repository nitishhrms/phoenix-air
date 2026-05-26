"""
Phoenix Air chatbot test suite.

Covers three retrieval layers for both policy and general paths:

  L1 = keyword/rule-based  -> response_type: hardcoded  (no LLM)
  L2 = embedding-based     -> response_type: llm        (LLM grounded in retrieved doc)
  L3 = web search          -> response_type: llm        (general path only)
  L4 = LLM model knowledge -> response_type: llm        (last resort)

Policy path:  L1 keyword  -> L2 embedding (threshold 0.65) -> falls to general path
General path: L1 keyword  -> L2 embedding (threshold 0.45) -> L3 web search -> L4 LLM

Note: If sentence_transformers is unavailable, L2 falls back to TF-IDF retrieval.

Usage: python test_suite.py
"""

import requests
import uuid

BASE = "http://localhost:8000/api/voice"
results = []


def call(session_id, text, caller_phone="+15550000001"):
    r = requests.post(BASE, json={
        "text":         text,
        "session_id":   session_id,
        "caller_phone": caller_phone,
        "language":     "en",
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def sid():
    return str(uuid.uuid4())[:8]


def test(label, resp, contains=None, rtype=None, not_contains=None, show=False):
    response = resp.get("response", "")
    rt = resp.get("response_type", "")
    ok = True
    reason = ""

    if contains and contains.lower() not in response.lower():
        ok = False
        reason = "expected '%s' in response" % contains
    if not_contains and not_contains.lower() in response.lower():
        ok = False
        reason = "did NOT expect '%s' in response" % not_contains
    if rtype and rt != rtype:
        ok = False
        reason = "expected rtype=%r, got %r" % (rtype, rt)

    mark = "PASS" if ok else "FAIL"
    print("  %s  %s" % (mark, label))
    if not ok or show:
        print("         rtype   : %s" % rt)
        print("         response: %s" % response[:140])
    if not ok and reason:
        print("         reason  : %s" % reason)
    results.append((label, ok))
    return ok


def section(title):
    print("\n%s" % title)
    print("-" * len(title))


# ============================================================
# POLICY PATH
# ============================================================

section("POLICY L1 -- keyword/rule-based (no LLM, response_type=hardcoded)")
# Each query contains an exact substring from _KEYWORD_MAP in rag.py.
# Answer is pre-authored in _POLICY_SHORT_ANSWERS — no LLM call at all.

s = sid(); r = call(s, "I need to cancel my flight — what is the cancellation fee?")
test("cancel my flight", r, rtype="hardcoded", contains="refund")

s = sid(); r = call(s, "What is the refund policy if I cancel my ticket?")
test("refund policy", r, rtype="hardcoded", contains="refund")

s = sid(); r = call(s, "Can I change my flight to a different date?")
test("change my flight", r, rtype="hardcoded", contains="change")

s = sid(); r = call(s, "What is the baggage allowance for a carry-on?")
test("carry-on baggage allowance", r, rtype="hardcoded", contains="carry")

s = sid(); r = call(s, "Is there a fee to pick my seat?")
test("seat selection fee", r, rtype="hardcoded", contains="seat")

s = sid(); r = call(s, "When does online check-in open and close?")
test("check-in deadline", r, rtype="hardcoded", contains="check-in")

s = sid(); r = call(s, "Can I bring my dog on the plane?")
test("pet policy (dog)", r, rtype="hardcoded", contains="pet")

s = sid(); r = call(s, "How do I earn miles on Phoenix Air?")
test("loyalty miles", r, rtype="hardcoded", contains="mile")

s = sid(); r = call(s, "Do you serve halal meals on board?")
test("special meal (halal)", r, rtype="hardcoded", contains="meal")

s = sid(); r = call(s, "Can I pay with PayPal?")
test("payment method (PayPal)", r, rtype="hardcoded", contains="paypal")

s = sid(); r = call(s, "My bag was lost at the airport, what do I do?")
test("lost bag", r, rtype="hardcoded", contains="baggage")

s = sid(); r = call(s, "My flight is delayed by 4 hours — what compensation do I get?")
test("flight delayed compensation", r, rtype="hardcoded", contains="delay")


section("POLICY L2 -- embedding-based (LLM grounded in retrieved policy doc)")
# These queries avoid exact L1 keywords but embed semantically close to policy docs.
# The system retrieves the relevant policy doc and generates an LLM answer.

s = sid(); r = call(s, "If I have a family emergency and absolutely cannot travel, what happens to my ticket?")
test("cancellation (no 'cancel' keyword)", r, rtype="llm", contains="refund")

s = sid(); r = call(s, "I want to bring my acoustic guitar on the plane — are there additional charges?")
test("instrument as baggage (no keyword)", r, rtype="llm")

s = sid(); r = call(s, "My suitcase came out of baggage claim with a crack in it. What is the process?")
test("damaged suitcase (baggage claim = L1 keyword)", r, rtype="hardcoded", contains="bag")

s = sid(); r = call(s, "I'm an expectant mother — do airlines have any rules about flying while pregnant?")
test("pregnant travel (no keyword -> policy L2 embedding)", r, rtype="llm", not_contains="not able to help")

s = sid(); r = call(s, "My mother needs extra time boarding and cannot walk long distances. What help is available?")
test("mobility assistance (no 'wheelchair' keyword)", r, rtype="llm")

s = sid(); r = call(s, "If Phoenix Air cancels my flight due to a storm, am I entitled to hotel accommodation?")
test("weather cancellation compensation", r, rtype="llm")


section("POLICY L3/L4 -- LLM from model/web (policy doc miss -> general fallback)")
# These are airline policy questions outside Phoenix Air's specific docs.
# Policy L1 and L2 miss -> falls through to general path -> web search or LLM model knowledge.

s = sid(); r = call(s, "What are the general rules for traveling with a pacemaker on a commercial flight?")
test("pacemaker travel (general/LLM)", r, rtype="llm", not_contains="not able to help")

s = sid(); r = call(s, "Is there typically a grace period if I'm a few minutes late for boarding?")
test("boarding grace period (LLM fallback)", r, rtype="llm", not_contains="not able to help")


# ============================================================
# GENERAL PATH
# ============================================================

section("GENERAL L1 -- keyword/rule-based (no LLM, response_type=hardcoded)")
# Each query contains an exact substring from _GENERAL_RULES in rag.py.
# Pre-authored answer returned directly — no LLM call.

s = sid(); r = call(s, "Is there wifi on the plane?")
test("wifi (keyword='wifi')", r, rtype="hardcoded", not_contains="policy")

s = sid(); r = call(s, "Do you have movies on board?")
test("movies (keyword='movies on board')", r, rtype="hardcoded")

s = sid(); r = call(s, "How early should I arrive at the airport?")
test("arrive early (keyword='arrive at the airport')", r, rtype="hardcoded", contains="hour")

s = sid(); r = call(s, "Can I track my flight?")
test("track flight (keyword='track my flight')", r, rtype="hardcoded")

s = sid(); r = call(s, "What happens if my flight is delayed?")
test("flight delayed (keyword='flight delayed')", r, rtype="hardcoded", contains="delay")

s = sid(); r = call(s, "How does the boarding process work?")
test("boarding process (keyword='boarding process')", r, rtype="hardcoded")

s = sid(); r = call(s, "How long do I need for a connecting flight?")
test("connecting flight (keyword='connecting flight')", r, rtype="hardcoded")

s = sid(); r = call(s, "Do I need a passport for international travel?")
test("passport (keyword='passport')", r, rtype="hardcoded")

s = sid(); r = call(s, "When is the cheapest time to book a flight?")
test("cheapest booking time (keyword='cheapest time to book')", r, rtype="hardcoded")


section("GENERAL L2 -- embedding-based (LLM grounded in retrieved general doc)")
# These avoid L1 keywords but embed close to a general knowledge doc (threshold 0.45).
# Covers docs: entertainment, security, boarding, connecting_flights, flight_status,
#              international_travel, booking_tips, airport_arrival.

s = sid(); r = call(s, "Are there USB sockets to charge my phone during the flight?")
test("USB charging (no keyword -> entertainment_doc)", r, rtype="llm", not_contains="not able to help")

s = sid(); r = call(s, "What items am I not allowed to bring through the airport scanner?")
test("prohibited items (no keyword -> security_doc)", r, rtype="llm")

s = sid(); r = call(s, "When is the gate cutoff — how late can I show up before I miss the flight?")
test("gate cutoff time (no keyword -> boarding_doc)", r, rtype="llm", contains="gate")

s = sid(); r = call(s, "If I have a brief stop at a hub airport, do my bags automatically go to my final destination?")
test("bag transfer at hub (no 'layover' keyword -> connecting_flights_doc)", r, rtype="llm", contains="bag")

s = sid(); r = call(s, "How can I check whether my departure is running on schedule?")
test("flight schedule check (no keyword -> flight_status_doc)", r, rtype="llm")

s = sid(); r = call(s, "What do I need to declare when going through customs on arrival?")
test("customs declaration (no keyword -> international_travel_doc)", r, rtype="llm")

s = sid(); r = call(s, "Is flying on a Tuesday or Wednesday actually cheaper than the weekend?")
test("cheap travel days (no keyword -> booking_tips_doc)", r, rtype="llm")

s = sid(); r = call(s, "Is 90 minutes enough time at the airport if I only have a carry-on?")
test("90-min arrival with carry-on (carry-on = L1 keyword -> hardcoded)", r, rtype="hardcoded")

s = sid(); r = call(s, "Is two hours typically enough to get through the terminal before a domestic flight?")
test("2-hour domestic buffer (no keyword -> airport_arrival_doc L2)", r, rtype="llm", contains="hour")


section("GENERAL L3/L4 -- web search or LLM model knowledge")
# These fall through both general L1 and L2 (no doc match).
# L3: web search results used as context.
# L4: LLM answers from training knowledge.

s = sid(); r = call(s, "How long does jet lag typically last after a long-haul flight?")
test("jet lag duration (web/LLM)", r, rtype="llm", not_contains="not able to help")

s = sid(); r = call(s, "Do commercial aircraft have defibrillators on board in case of cardiac arrest?")
test("defibrillator on planes (web/LLM)", r, rtype="llm", not_contains="not able to help")

s = sid(); r = call(s, "Why do airlines dim the cabin lights during takeoff and landing?")
test("cabin lights dimming (web/LLM)", r, rtype="llm", not_contains="not able to help")


# ============================================================
# GREETING & OOD
# ============================================================

section("GREETING -- answered at any booking stage")

s = sid(); r = call(s, "hello")
test("hello at IDLE", r, not_contains="not able to help")

s = sid(); r = call(s, "hi there, how are you?")
test("hi there at IDLE", r, not_contains="not able to help")

# Greeting mid-booking
s = sid()
call(s, "I want to book a flight")   # start booking -> COLLECTING_PHONE
r = call(s, "hey, quick question first — is there wifi?")
test("wifi question mid-booking", r, rtype="hardcoded", not_contains="not able to help")


section("OUT-OF-DOMAIN -- declined politely")

s = sid(); r = call(s, "what is the capital of France?")
test("geography OOD", r, contains="airline")

s = sid(); r = call(s, "tell me a joke")
test("joke OOD", r, contains="airline")

s = sid(); r = call(s, "write me a poem")
test("poem OOD", r, contains="airline")


# ============================================================
# ANSWER VALIDATION (booking flow)
# ============================================================

section("BOOKING FLOW & ANSWER VALIDATION")

s = sid(); r = call(s, "I want to book a flight")
test("start booking", r, not_contains="not able to help")
print("         conv_state: %s" % r.get("conv_state", ""))

# Policy question mid-booking (should not break the flow)
s = sid()
call(s, "book a flight")
r = call(s, "what is the baggage allowance?")
test("policy Q mid-booking (no 'not able to help')", r, not_contains="not able to help")
print("         response: %s" % r.get("response", "")[:120])

# ============================================================
# SUMMARY
# ============================================================
passed = sum(1 for _, ok in results if ok)
total = len(results)
print("\n" + "=" * 60)
print("  Results: %d/%d passed" % (passed, total))
if passed == total:
    print("  ALL TESTS PASSED")
else:
    failed = [lbl for lbl, ok in results if not ok]
    print("  FAILED:")
    for f in failed:
        print("    - %s" % f)
print("=" * 60 + "\n")
