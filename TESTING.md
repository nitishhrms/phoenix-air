# Phoenix Air — Testing Guide

---

## Policy & General QA — Architecture

### What Was Built

Two separate pipelines handle non-booking questions. Both live in `nodes/rag_policy.py` and `services/rag.py`.

---

### Pipeline Order (priority top → bottom)

```
User asks a question
        │
        ▼
 Intent Router  ──────────────────────────────────────────────────
 (intent_router.py)                                               │
  Layer 1: keyword match (POLICY_KEYWORDS / GENERAL_QA_KEYWORDS) │
  Layer 2: embedding similarity vs anchor sentences              │
  Layer 3: LLM classifier (Haiku)                                │
        │                                                         │
        ▼                                                         ▼
 intent = "policy"                                    intent = "out_of_domain"
 intent = "general_qa"                                        │
        │                                              Fixed hardcoded decline
        ▼                                              (no LLM, instant)
 rag_policy_node
 (nodes/rag_policy.py)
        │
        ├─► POLICY L1 — keyword match → _KEYWORD_MAP
        │     Match:  instant hardcoded short answer (no LLM)
        │     No match: continue ↓
        │
        ├─► POLICY L2 — embedding cosine similarity
        │     Model:     all-MiniLM-L6-v2 (sentence-transformers)
        │     Matches:   query vs 140 seed questions + 14 policy docs
        │     Method:    dual-match — max(seed_score, doc_score) per policy
        │     Threshold: 0.65  (false positives rejected by hallucination guard)
        │     On hit:    LLM grounded to policy doc → hallucination guard check
        │     Guard fail: fall through ↓  (wrong doc — don't show bad answer)
        │     Fallback:  TF-IDF (sklearn) if sentence-transformers unavailable
        │     No match:  continue ↓
        │
        ├─► GENERAL L1 — keyword match → _GENERAL_RULES
        │     10 topic rules: wifi, entertainment, security, boarding,
        │     connections, flight tracking, arrival time, booking tips,
        │     international travel, travel insurance
        │     Match:  instant hardcoded answer (no LLM)
        │     No match: continue ↓
        │
        ├─► GENERAL L2 — embedding cosine similarity on general docs
        │     8 general knowledge docs: entertainment, security, boarding,
        │     connecting flights, flight status, international travel,
        │     booking tips, airport arrival
        │     Threshold: 0.45
        │     On hit:    LLM grounded to general doc
        │     Fallback:  TF-IDF if sentence-transformers unavailable
        │     No match:  continue ↓
        │
        ├─► GENERAL L3 — DuckDuckGo web search → LLM
        │     For real-world questions not covered in docs
        │     No results: continue ↓
        │
        ├─► GENERAL L4 — LLM from model knowledge
        │     Haiku answers from training knowledge
        │     No answer: continue ↓
        │
        └─► GENERAL L5 — Escalate
              Sets intent = customer_contact
              Gives support email + phone number
```

---

### Key Files

| File | What it does |
|---|---|
| `nodes/intent_router.py` | Classifies intent: policy / general_qa / booking / OOD / etc. |
| `nodes/rag_policy.py` | Runs the full policy + general pipeline in order |
| `services/rag.py` | All retrieval logic: keyword maps, policy docs, general docs, embedding matrices, TF-IDF |
| `services/embedder.py` | Shared `all-MiniLM-L6-v2` singleton — loaded once at startup |
| `nodes/out_of_domain.py` | Fixed hardcoded OOD response, zero LLM |

---

### Policy Data (services/rag.py)

**14 policy documents** with pre-authored short answers (L1) and full text for LLM grounding (L2):

| Policy ID | Topic |
|---|---|
| `refund_policy` | Refunds, 24hr window, $75 fee |
| `cancellation_policy` | Passenger cancellations, no-show |
| `change_policy` | Flight changes, fees by notice period |
| `baggage_policy` | Carry-on, checked bag fees, oversize |
| `seat_policy` | Seat selection, upgrades |
| `checkin_policy` | Online/airport check-in deadlines |
| `pet_policy` | Pets in cabin/hold, service animals |
| `special_assistance_policy` | Wheelchair, oxygen, pre-boarding |
| `infant_child_policy` | Lap infants, unaccompanied minors |
| `loyalty_policy` | Miles, tiers, redemption |
| `delay_cancellation_policy` | Airline-caused delays/cancellations |
| `lost_baggage_policy` | Lost/damaged bags, compensation |
| `meal_policy` | In-flight food, special meals |
| `payment_policy` | Accepted payment methods |

Each policy has **10–12 seed questions** used for embedding retrieval.

---

### General Knowledge Data (services/rag.py)

**8 general airline knowledge documents** (not Phoenix Air specific):

| Doc ID | Topic |
|---|---|
| `entertainment_doc` | Movies, screens, Wi-Fi, charging |
| `security_doc` | TSA rules, liquids, laptops |
| `boarding_doc` | Boarding process, groups, gate close |
| `connecting_flights_doc` | Layovers, missed connections, bag transfer |
| `flight_status_doc` | Flight tracking, delay notifications |
| `international_travel_doc` | Passport, visa, customs, arrival |
| `booking_tips_doc` | Best time to book, saving money |
| `airport_arrival_doc` | How early to arrive, domestic vs international |

Each doc has **10–12 seed questions** for embedding retrieval.

---

### Thresholds

| Layer | Type | Threshold | Why |
|---|---|---|---|
| Policy L2 | Embedding | 0.65 | Real policy Q scores 0.80+; false positives ~0.55 |
| General L2 | Embedding | 0.45 | Broader topics, needs lower bar |
| Policy L2 | TF-IDF fallback | 0.25 | Only when sentence-transformers unavailable |
| General L2 | TF-IDF fallback | 0.12 | Lower bar for broader general topics |

---

### What to Test for Each Layer

#### Policy L1 — Keyword Match
**How to verify:** `response_type` must be `hardcoded`, answer is instant (< 1s).  
Test with exact and natural phrasing:
```
"my bag was lost"                          → lost_baggage_policy
"what if my flight gets cancelled"         → delay_cancellation_policy
"what is your refund policy"               → refund_policy
"how much to change my flight"             → change_policy
"do you accept PayPal"                     → payment_policy
"can I bring my dog"                       → pet_policy
"what is the baggage allowance"            → baggage_policy
"when does check-in open"                  → checkin_policy
"can I choose my seat"                     → seat_policy
"do you have a loyalty program"            → loyalty_policy
"can I travel with my infant"              → infant_child_policy
"can I get wheelchair assistance"          → special_assistance_policy
"what meals are served"                    → meal_policy
```

#### Policy L2 — Embedding Match
**How to verify:** `response_type` is `llm`, answer is grounded to a policy doc.  
Test with paraphrased/indirect policy questions (don't use exact L1 keywords):
```
"I'd like to understand your cancellation terms"
"tell me about your luggage rules"
"what happens to my ticket if I need to cancel"
"is there any fee if I need to rebook"
"what are the rules around travelling with animals"
```

#### General L1 — Keyword Match
**How to verify:** `response_type` must be `hardcoded`.  
```
"is there wifi on the plane"
"do you have movies on the flight"
"what are the airport security rules"
"how does boarding work"
"how do I track my flight"
"what happens if I miss my connecting flight"
"how early should I get to the airport"
"when is the cheapest time to book"
"do I need travel insurance"
```

#### General L2 — Embedding Match
**How to verify:** `response_type` is `llm`, answer references specific travel knowledge.  
```
"what documents do I need for international travel"
"can I leave the airport during a long layover"
"how far in advance should I book for international"
"is 1 hour enough for a domestic connection"
"what should I know about flying internationally for the first time"
```

#### OOD — Out of Domain
**How to verify:** Fixed response about specialising in flight booking. No LLM called.  
```
"what is the capital of France"
"tell me a joke"
"who won the World Cup"
"how do I cook pasta"
"recommend a Netflix show"
```

#### Mid-Booking Policy/General (important edge case)
At `COLLECTING_DEPARTURE` state, send a policy question — agent should answer it and **stay at the same booking state**, not advance the booking:
```
1. Auth → COLLECTING_DEPARTURE
2. Send: "what is your refund policy"   → answers policy, still at COLLECTING_DEPARTURE
3. Send: "New York"                     → booking continues to COLLECTING_DESTINATION
```

---

## 1. Starting the Server

```bash
# Activate virtual environment first
venv\Scripts\activate

# Start server
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Wait for both lines to appear:
```
[STARTUP] Embedding warmup complete.
INFO:     Application startup complete.
```
Warmup takes ~15-30 seconds on first run (builds embedding matrices).

---

## 2. API Endpoint

All conversation goes through one endpoint:

```
POST http://localhost:8000/api/voice
```

**Request body:**
```json
{
  "text": "<user message>",
  "session_id": "<any unique string>",
  "caller_phone": ""
}
```

**Response fields:**
```json
{
  "response": "<agent reply>",
  "conv_state": "<current state>",
  "response_type": "hardcoded | llm",
  "end_call": false,
  "suggestions": [],
  "flights": []
}
```

`response_type: "hardcoded"` = instant rule-based answer (no LLM called)  
`response_type: "llm"` = LLM was used

---

## 3. Auth Flow (required before every test)

Run these 4 calls in order with the same `session_id`:

| Step | text | Expected state |
|---|---|---|
| 1 | `""` (empty) | `SELECTING_LANGUAGE` |
| 2 | `"1"` (English) | `COLLECTING_PHONE` |
| 3 | `"+1555xxxxxxx"` | `VERIFYING_OTP` — OTP printed in response |
| 4 | `"<6-digit OTP>"` | `COLLECTING_DEPARTURE` |

---

## 4. Booking Flow

After auth, go through the booking states in order:

| State | What to send | Expected next state |
|---|---|---|
| `COLLECTING_DEPARTURE` | city name e.g. `"New York"` | `COLLECTING_DESTINATION` |
| `COLLECTING_DESTINATION` | city name e.g. `"London"` | `COLLECTING_DATE` |
| `COLLECTING_DATE` | date e.g. `"August 15 2026"` | `SELECTING_FLIGHT` |
| `SELECTING_FLIGHT` | `"1"` or `"2"` or `"3"` | `COLLECTING_PASSENGER` |
| `COLLECTING_PASSENGER` | first + last name e.g. `"John Smith"` | `COLLECTING_CONTACT` |
| `COLLECTING_CONTACT` | email or phone | `CONFIRMING_BOOKING` |
| `CONFIRMING_BOOKING` | `"yes"` or `"confirm"` | `POST_BOOKING` |

**Date formats accepted:** `August 15 2026` / `Aug 15 2026` / `2026-08-15` / `15 August 2026`

**Edge cases to test:**
- Same city for departure and destination → agent should reject it
- Invalid date e.g. `"tomorrow"` → agent asks for correct format
- City not in database → agent suggests nearby cities

---

## 5. Policy Questions (L1 — instant hardcoded, no LLM)

These should always return `response_type: "hardcoded"`:

| Question | Expected policy |
|---|---|
| `"my bag was lost"` | Lost baggage — compensation amounts |
| `"what if my flight gets cancelled by the airline"` | Delay/cancellation — refund or rebooking |
| `"what is the refund policy"` | Refund — 24hr window, $75 fee |
| `"how much does it cost to change my flight"` | Change fee — free 7+ days, $35/$50 |
| `"can I bring my dog on the plane"` | Pet policy — $95 cabin, 20lb limit |
| `"do you accept PayPal"` | Payment — all accepted methods |
| `"what is the baggage allowance"` | Baggage — carry-on free, checked $35+ |
| `"when does check-in open"` | Check-in — 24hr online, 45min airport |
| `"can I choose my seat"` | Seat — free standard, $25-50 preferred |
| `"do you have a loyalty program"` | Loyalty — miles, tiers, redemption |
| `"can I travel with my infant"` | Infant — free lap infant domestic |
| `"can I get wheelchair assistance"` | Special assistance — 48hr notice |
| `"what meals are served"` | Meal — snacks 90min+, hot meal 4hr+ |
| `"will I get a refund if the airline cancels"` | Delay/cancellation — full refund |

---

## 6. General QA Questions (L1 — instant hardcoded)

These should return `response_type: "hardcoded"`:

| Question | Expected topic |
|---|---|
| `"is there wifi on the plane"` | Wi-Fi availability |
| `"do you have movies on the plane"` | In-flight entertainment |
| `"what are the airport security rules"` | TSA / liquids / laptops |
| `"how does boarding work"` | Boarding order and timing |
| `"how do I track my flight"` | Flight status / FlightAware |
| `"what happens if I miss my connecting flight"` | Connection rebooking |
| `"how early should I get to the airport"` | 2hr domestic, 3hr international |
| `"when is the cheapest time to book a flight"` | Booking tips |
| `"do I need travel insurance"` | Insurance overview |

---

## 7. General QA Questions (L2 — LLM grounded to general doc)

These return `response_type: "llm"` with a detailed, doc-grounded answer:

| Question | Expected topic |
|---|---|
| `"what documents do I need for international travel"` | Passport/visa requirements |
| `"can I leave the airport during a long layover"` | Layover / transit rules |
| `"how far in advance should I book internationally"` | Booking tips doc |
| `"is 1 hour enough time for my connection"` | Connecting flights |

---

## 8. Out-of-Domain Questions

These should return a fixed decline message (no LLM):

| Question | Expected behaviour |
|---|---|
| `"what is the capital of France"` | Fixed OOD response |
| `"tell me a joke"` | Fixed OOD response |
| `"who won the World Cup"` | Fixed OOD response |
| `"how do I cook pasta"` | Fixed OOD response |

---

## 9. Mid-Booking Policy/General Questions

At any state (e.g. `COLLECTING_DEPARTURE`), you can ask a policy or general question and then continue booking. Example:

```
1. Auth → COLLECTING_DEPARTURE
2. "what is your refund policy"   → hardcoded answer, stays at COLLECTING_DEPARTURE
3. "New York"                     → confirms departure, moves to COLLECTING_DESTINATION
```

---

## 10. Post-Booking Questions

After confirming a booking, `conv_state` becomes `POST_BOOKING`. You can ask further questions:

```
1. Complete booking → POST_BOOKING
2. "can I change my flight"  → policy answer
3. "is there wifi"           → general answer
4. "book another flight"     → restarts the booking flow
```

---

## 11. Response Type Reference

| response_type | Meaning | LLM called? |
|---|---|---|
| `hardcoded` | Keyword matched — instant pre-written answer | No |
| `llm` | LLM generated answer (may be policy-grounded or general) | Yes |

---

## 12. Quick PowerShell Test Script

```powershell
$s = "test_$(Get-Random)"
$base = "http://localhost:8000/api/voice"

function Send($text) {
    $body = "{`"text`":`"$text`",`"session_id`":`"$s`",`"caller_phone`":`"`"}"
    Invoke-RestMethod -Uri $base -Method POST -ContentType "application/json" -Body $body
}

# Auth
Send "" | Out-Null
Send "1" | Out-Null
$r = Send "+15551234567"
$otp = ($r.response | Select-String '\d{6}').Matches[0].Value
Send $otp | Out-Null

# Test a question
$r = Send "what is the refund policy"
Write-Host "Type: $($r.response_type)"
Write-Host "Answer: $($r.response)"
```
