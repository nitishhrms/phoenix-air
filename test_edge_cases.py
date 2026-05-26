# Edge case test suite - runs against the live server on http://localhost:8000
# Usage: python test_edge_cases.py

import httpx
import uuid
import sys

BASE = "http://localhost:8000/api/voice"
PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
HEAD = "\033[94m"
RESET = "\033[0m"

results = []


def sid():
    return str(uuid.uuid4())


def call(text, session_id, caller_phone=""):
    r = httpx.post(BASE, json={"text": text, "session_id": session_id,
                                "caller_phone": caller_phone}, timeout=60)
    r.raise_for_status()
    return r.json()


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(condition)
    tag = f"{HEAD}[{name}]{RESET}"
    print(f"  {status} {tag} {detail}")
    return condition


def section(title):
    print(f"\n{HEAD}{'-'*60}{RESET}")
    print(f"{HEAD}  {title}{RESET}")
    print(f"{HEAD}{'-'*60}{RESET}")


# ------------------------------------------------------------------
section("TEST 1 - Normal full booking (US phone -> SMS)")
# ------------------------------------------------------------------
s = sid()
r = call("", s)
check("greeting received",    "Phoenix Air" in r["response"] or "departing" in r["response"].lower())
check("conv_state=COLLECTING_DEPARTURE", r["conv_state"] == "COLLECTING_DEPARTURE")

r = call("Los Angeles", s)
check("departure resolved to LAX", "LAX" in r["response"])
check("conv_state=COLLECTING_DESTINATION", r["conv_state"] == "COLLECTING_DESTINATION")

r = call("New York", s)
check("destination resolved to JFK", "JFK" in r["response"])
check("conv_state=COLLECTING_DATE", r["conv_state"] == "COLLECTING_DATE")

r = call("August 15 2026", s)
check("flights returned",       "Option 1" in r["response"])
check("conv_state=PRESENTING",  r["conv_state"] == "PRESENTING_FLIGHTS")
check("end_call=False",         r["end_call"] == False)

r = call("option 1", s)
check("flight selected",        "Could I get your full name" in r["response"] or "full name" in r["response"].lower())
check("conv_state=COLLECTING_PASSENGER", r["conv_state"] == "COLLECTING_PASSENGER")

r = call("Jane Doe", s)
check("name accepted",          "Jane Doe" in r["response"] or "Jane" in r["response"])
check("conv_state=COLLECTING_CONTACT", r["conv_state"] == "COLLECTING_CONTACT")

r = call("+14085551234", s)
check("booking confirmed",      r["conv_state"] == "BOOKING_CONFIRMED")
check("confirmation number",    "PHN-" in r["response"] or "CONF" in r["response"])
check("end_call=True",          r["end_call"] == True)
check("transfer=False",         r["transfer"] == False)


# ------------------------------------------------------------------
section("TEST 2 - No flights available (AAL -> YVR, 404)")
# ------------------------------------------------------------------
s = sid()
call("", s)
call("Aalborg", s)
call("Vancouver", s)
r = call("September 1 2026", s)
check("no flights message",  "no flights" in r["response"].lower() or "not available" in r["response"].lower())
check("end_call=True",       r["end_call"] == True)
check("conv_state=DONE",     r["conv_state"] == "DONE")


# ------------------------------------------------------------------
section("TEST 3 - Unknown city (retry up to 2x, then end call)")
# ------------------------------------------------------------------
s = sid()
call("", s)
r = call("Gotham City", s)
check("retry prompt",        "couldn't find" in r["response"].lower() or "try" in r["response"].lower())
check("still collecting dep",r["conv_state"] == "COLLECTING_DEPARTURE")
check("end_call=False",      r["end_call"] == False)

r = call("Gotham City again", s)
check("max retries: end call","call back" in r["response"].lower() or "sorry" in r["response"].lower())
check("end_call=True",        r["end_call"] == True)


# ------------------------------------------------------------------
section("TEST 4 - Invalid date (past date)")
# ------------------------------------------------------------------
s = sid()
call("", s)
call("Los Angeles", s)
call("New York", s)
r = call("January 1 2020", s)
check("past date rejected",  "past" in r["response"].lower() or "future" in r["response"].lower() or "valid" in r["response"].lower())
check("stays on COLLECTING_DATE", r["conv_state"] == "COLLECTING_DATE")
check("end_call=False",      r["end_call"] == False)


# ------------------------------------------------------------------
section("TEST 5 - Date too far in future (>1 year)")
# ------------------------------------------------------------------
s = sid()
call("", s)
call("Los Angeles", s)
call("New York", s)
r = call("January 1 2035", s)
check("far-future date rejected", "year" in r["response"].lower() or "advance" in r["response"].lower())
check("stays on COLLECTING_DATE", r["conv_state"] == "COLLECTING_DATE")


# ------------------------------------------------------------------
section("TEST 6 - Transfer to customer support (any state)")
# ------------------------------------------------------------------
# Mid-flow transfer
s = sid()
call("", s)
call("Los Angeles", s)
r = call("I want to speak to a customer support agent", s)
check("transfer triggered",  r["transfer"] == True)
check("end_call=True",       r["end_call"] == True)
check("hold message",        "transfer" in r["response"].lower() or "hold" in r["response"].lower())

# Transfer from IDLE
s2 = sid()
call("", s2)
r2 = call("transfer me to a human", s2)
check("transfer from IDLE",  r2["transfer"] == True)


# ------------------------------------------------------------------
section("TEST 7 - Policy RAG (mid-flow, should resume same state)")
# ------------------------------------------------------------------
s = sid()
call("", s)
call("Los Angeles", s)
r_before = call("New York", s)
check("state before policy Q", r_before["conv_state"] == "COLLECTING_DATE")

r = call("What is your refund policy?", s)
check("policy answered",     "refund" in r["response"].lower() or "24 hours" in r["response"].lower())
check("conv_state unchanged",r["conv_state"] == "COLLECTING_DATE")  # resumes same state
check("end_call=False",      r["end_call"] == False)

# Follow-up: booking continues after policy answer
r2 = call("August 20 2026", s)
check("booking resumes after policy", r2["conv_state"] == "PRESENTING_FLIGHTS")


# ------------------------------------------------------------------
section("TEST 8 - Cancellation policy query")
# ------------------------------------------------------------------
s = sid()
call("", s)
r = call("What happens if I need to cancel my flight?", s)
check("cancel policy answered", "cancel" in r["response"].lower() or "fee" in r["response"].lower())


# ------------------------------------------------------------------
section("TEST 9 - Non-US contact (email -> email notification)")
# ------------------------------------------------------------------
s = sid()
call("", s)
call("San Francisco", s)
call("Los Angeles", s)
call("October 10 2026", s)
call("option 1", s)
call("John Smith", s)
r = call("john.smith@example.com", s)
check("booking confirmed",    r["conv_state"] == "BOOKING_CONFIRMED")
check("email method in resp", "email" in r["response"].lower())
check("end_call=True",        r["end_call"] == True)


# ------------------------------------------------------------------
section("TEST 10 - Same departure and destination")
# ------------------------------------------------------------------
s = sid()
call("", s)
call("Los Angeles", s)
r = call("Los Angeles", s)   # same as departure
check("same-city rejected",  "same" in r["response"].lower() or "can't" in r["response"].lower() or "cannot" in r["response"].lower())
check("stays on COLLECTING_DESTINATION", r["conv_state"] == "COLLECTING_DESTINATION")


# ------------------------------------------------------------------
section("TEST 11 - Restart mid-flow")
# ------------------------------------------------------------------
s = sid()
call("", s)
call("Los Angeles", s)
call("New York", s)
r = call("start over", s)
check("restart acknowledged", "fresh" in r["response"].lower() or "start" in r["response"].lower() or "departing" in r["response"].lower())
check("state reset to COLLECTING_DEPARTURE", r["conv_state"] == "COLLECTING_DEPARTURE")


# ------------------------------------------------------------------
section("TEST 12 - Returning user (cross-call memory)")
# ------------------------------------------------------------------
phone = "+19175550199"
s1 = sid()
# First call - complete booking
call("", s1, phone)
call("Dallas", s1, phone)
call("Miami", s1, phone)
call("November 5 2026", s1, phone)
call("option 1", s1, phone)
call("Alice Walker", s1, phone)
call("+19175550199", s1, phone)

# Second call - should greet by name
s2 = sid()
r = call("", s2, phone)
check("returning user greeted by name", "Alice" in r["response"] or "welcome back" in r["response"].lower())


# ------------------------------------------------------------------
section("TEST 13 - Flight selection by airline name")
# ------------------------------------------------------------------
s = sid()
call("", s)
call("Boston", s)
call("Chicago", s)
r = call("September 15 2026", s)
# Get the first airline name from the response
import re
airlines = re.findall(r'(Delta|United|Alaska|Southwest|JetBlue|American)', r["response"])
if airlines:
    r2 = call(airlines[0], s)
    check("select by airline name", r2["conv_state"] == "COLLECTING_PASSENGER")
else:
    check("flights listed (need airline name)", False, "(no airline found in response)")


# ------------------------------------------------------------------
print(f"\n{'-'*60}")
passed = sum(results)
total  = len(results)
color  = "\033[92m" if passed == total else "\033[93m"
print(f"{color}  Results: {passed}/{total} checks passed{RESET}")
print(f"{'-'*60}\n")
sys.exit(0 if passed == total else 1)
