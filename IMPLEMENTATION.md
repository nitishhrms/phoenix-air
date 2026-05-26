# Phoenix Air — Implementation Details

## 1. System Overview

A voice/text airline booking agent built on FastAPI + LangGraph. A user chats (or calls via Phonely), the backend processes each turn through a graph of nodes, and returns a spoken/text response.

```
Browser / Phonely
      │  POST /api/voice { text, session_id, caller_phone }
      ▼
FastAPI (main.py)
  → load session from SQLite
  → run LangGraph graph (15s timeout)
  → save session to SQLite
  → return { response, end_call, transfer, conv_state, flights }
      │
      ▼
LangGraph Graph (graph.py)
  intent_router → should_continue() → action node → END
```

---

## 2. Conversation States

```
COLLECTING_PHONE      → ask for phone number (OTP auth)
VERIFYING_OTP         → ask for 6-digit code
COLLECTING_DEPARTURE  → "Which city are you departing from?"
COLLECTING_DESTINATION→ "Where would you like to fly to?"
COLLECTING_DATE       → "What date would you like to travel?"
PRESENTING_FLIGHTS    → flight cards shown, waiting for selection
COLLECTING_PASSENGER  → "What is your full name?"
COLLECTING_CONTACT    → "Phone or email for confirmation?"
COLLECTING_PAYMENT    → "Shall I process payment of $X?"
BOOKING_CONFIRMED     → booking done, PDF sent, session cleaned up
DONE                  → end_call = True
```

---

## 3. Hybrid Intent Router (`nodes/intent_router.py`)

Runs every turn. Sets `state["intent"]` which controls routing in `should_continue()`.

### Layer 1 — Keyword scan (fastest)
Exact substring match against hardcoded keyword lists:

| Intent | Example keywords |
|--------|-----------------|
| `transfer` | "agent", "human", "speak to someone" |
| `policy` | "refund", "cancel", "baggage", "fees" |
| `restart` | "start over", "book another" |
| `greeting` | "hello", "hi", "good morning" |
| `field_correction` | "change my departure", "I meant", "update date" |
| `customer_contact` | "complaint", "billing issue", "lost bag" |
| `out_of_domain` | OOD triggers without any airline terms |

**Auth bypass:** During `COLLECTING_PHONE` / `VERIFYING_OTP` states, all 3 layers are skipped and intent is always `normal` — so phone numbers and OTP codes are never misclassified.

### Layer 2 — Sentence embedding (catches paraphrases)
Uses `all-MiniLM-L6-v2` (22MB, CPU-only) from `sentence-transformers`.
- Encodes user input and compares against pre-computed anchor vectors per intent
- Fires only if cosine similarity ≥ 0.45 and intent is non-normal
- Embeddings loaded once at startup (module scope), not per call

### Layer 3 — LLM fallback (Claude Haiku)
If layers 1 and 2 both return `normal`, asks Claude Haiku to classify into one of the known intents.
- Only fires for inputs longer than 3 characters
- Prompt: "Classify into exactly one of: {intents}. Reply with ONLY the intent name."

---

## 4. Graph Routing (`graph.py` → `should_continue()`)

```
Layer 1: Global interruptions (any state)
  transfer        → transfer node
  policy          → rag_policy node
  restart         → resolve_airport node (resets all fields)
  out_of_domain   → out_of_domain node
  greeting        → greeting node
  field_correction→ field_correction node
  customer_contact→ customer_contact node

Layer 2: State-driven routing (intent == normal)
  COLLECTING_PHONE/VERIFYING_OTP   → auth node
  COLLECTING_DEPARTURE/DESTINATION → resolve_airport node
  COLLECTING_DATE                  → search_flights node
  PRESENTING_FLIGHTS               → present_flights node
  COLLECTING_PASSENGER             → collect_passenger node
  COLLECTING_CONTACT               → book_flight node
  COLLECTING_PAYMENT               → payment node
  BOOKING_CONFIRMED/DONE           → end_node
```

---

## 5. All Nodes

| Node | File | What it does |
|------|------|-------------|
| `intent_router` | `nodes/intent_router.py` | 3-layer intent classification |
| `resolve_airport` | `nodes/resolve_airport.py` | City → IATA via SQLite + fuzzy match |
| `search_flights` | `nodes/search_flights.py` | Parse date, call API, rank flights |
| `present_flights` | `nodes/present_flights.py` | Parse "option 2" / "Delta" selection |
| `collect_passenger` | `nodes/collect_passenger.py` | Split first + last name |
| `book_flight` | `nodes/book_flight.py` | Call booking API, save to SQLite, → payment |
| `payment` | `nodes/payment.py` | Confirm charge, process mock payment, send PDF |
| `rag_policy` | `nodes/rag_policy.py` | LLM answer grounded in policy doc + hallucination guard |
| `transfer` | `nodes/transfer.py` | Set transfer=True, end_call=True |
| `out_of_domain` | `nodes/out_of_domain.py` | Politely decline off-topic questions |
| `greeting` | `nodes/greeting.py` | Context-aware greeting at any state |
| `customer_contact` | `nodes/customer_contact.py` | Support info + offer agent transfer |
| `field_correction` | `nodes/field_correction.py` | Reset field + downstream, re-ask |
| `auth` | `nodes/auth.py` | OTP phone verification |
| `end_node` | `graph.py` | Set end_call=True |

---

## 6. Services

### `services/rag.py` — Policy RAG
- `query_policy(question)` — keyword match → TF-IDF cosine similarity → returns raw policy doc
- `query_policy_llm(question)` — retrieves doc via above, then passes to Claude Haiku for a conversational answer grounded in that doc

### `services/hallucination_guard.py`
- `check_response(user_input, response, context) → (bool, str)`
- Calls Claude Haiku: "Reply PASS or FAIL: <reason>"
- Used after every LLM policy answer
- If guard itself errors → passes through (never blocks)

### `services/timeout_guard.py`
- `run_with_timeout(fn, args, timeout_s=15, retries=1)`
- Wraps `airline_graph.invoke()` in `main.py`
- On timeout: retries once, then returns fallback state with safe message

### `services/auth.py`
- `generate_otp(phone)` → 6-digit code stored in `otp_codes` table (10 min TTL)
- `verify_otp(phone, code)` → checks DB, marks verified, returns bool
- `send_otp(phone, code)` → Twilio SMS if configured, else console log

### `services/external_api.py`
- `search_flights(src, dst, date)` → GET AWS Lambda, returns `{flights: [...]}`
- `search_flights_multi(src, dst, dates)` → parallel calls via `ThreadPoolExecutor`
- `book_flight(...)` → POST AWS Lambda, returns confirmation

### `services/payment.py`
- `mock_charge(amount, contact, confirmation_number)` → always succeeds, logs to `payments` table, returns `{success: True, transaction_id: "TXN-XXXXXXXX"}`

### `services/pdf_ticket.py`
- `generate_ticket(booking)` → builds PDF via `reportlab`, returns bytes
- `send_ticket(contact, contact_type, pdf_bytes, booking)`:
  - Email: attaches PDF via SMTP (Gmail or any SMTP)
  - Phone: console log (mock MMS)

### `services/notifications.py`
- `send_confirmation(contact, contact_type, booking)` → booking confirmation SMS (Twilio) or email (console)

---

## 7. Date Parsing (`nodes/search_flights.py`)

`_parse_date(text)` tries three layers:

1. **Regex** — `MM/DD/YYYY`, `DD-MM-YYYY`, `15th August 2026`, ordinals
2. **dateutil** — `fuzzy=True` catches "August 15 2026", "next Friday", "tomorrow"
3. **LLM (Claude Haiku)** — final fallback: "Today is {date}. Extract travel date. Reply YYYY-MM-DD or UNKNOWN."

`_parse_multiple_dates(text)` detects:
- Range: "Aug 15 to Aug 17" → `["2026-08-15", "2026-08-16", "2026-08-17"]`
- List: "Aug 15 or 16" → `["2026-08-15", "2026-08-16"]`
- Max 3 dates, triggers parallel API search

---

## 8. Flight Ranking (`nodes/search_flights.py`)

`_rank_flights(flights)` scores each flight:
```
score = price + duration_minutes × 0.5
```
- Sorted ascending (lower = better value)
- Top flight gets `"recommended": True`
- Returned as sorted list; UI shows "Top Pick" badge on first card

---

## 9. Field Correction (`nodes/field_correction.py`)

Detects which field the user wants to change (keyword scan → LLM fallback), then resets that field **and all downstream fields**:

| Field changed | Fields cleared |
|--------------|----------------|
| departure | departure, destination, date, flights, passenger, contact, payment |
| destination | destination, date, flights, passenger, contact, payment |
| date | date, flights, passenger, contact, payment |
| passenger | passenger name, contact, payment |
| contact | contact, payment |

Sets `conv_state` back to the correct collection step.

---

## 10. Database Schema (`db/database.py`)

```sql
airports      -- IATA, city, name, aliases (static, seeded once)
sessions      -- session_id, conv_state, full state JSON, updated_at
user_profiles -- phone, first_name, last_name, last_from, last_to, call_count
bookings      -- confirmation, flight details, passenger, contact
otp_codes     -- phone, code, expires_at (10 min), verified flag
payments      -- transaction_id, confirmation_number, amount, contact, status
```

Session is **deleted** from `sessions` table after `BOOKING_CONFIRMED`.

---

## 11. Authentication Flow

```
User enters phone number
      │
auth_node (COLLECTING_PHONE)
  → validate phone format
  → generate_otp() → store in otp_codes
  → send_otp() → Twilio SMS or [OTP] console log
  → conv_state = VERIFYING_OTP

User enters 6-digit code
      │
auth_node (VERIFYING_OTP)
  → verify_otp() → check DB, check expiry
  → if valid: authenticated=True, conv_state=COLLECTING_DEPARTURE
  → if fail: retry_count++, max 3 attempts then end_call
```

---

## 12. Payment Flow

```
book_flight node
  → calls booking API (AWS Lambda POST)
  → saves booking to SQLite
  → sets conv_state = COLLECTING_PAYMENT

payment node
  → shows total: "Your total is $X. Confirm payment?"
  → user says "yes"
  → mock_charge() → logs to payments table
  → generate_ticket() → PDF bytes
  → send_ticket() → email attachment or console log
  → conv_state = BOOKING_CONFIRMED, end_call = True
```

---

## 13. Frontend (`frontend/index.html`)

Single-page HTML/CSS/JS — no framework, no build step.

### Flight Cards
When `conv_state == PRESENTING_FLIGHTS` and `data.flights.length > 0`:
- Renders clickable cards instead of text bubble
- "Top Pick" badge on recommended flight
- Sort bar: **Best Value** / **Price ↑** / **Time ↑** (client-side, instant)
- Multi-date: cards grouped under date headers
- Clicking "Select" sends `"option N"` to the API

### Step Sidebar
7 steps track booking progress. Steps go: done (green ✓) → active (blue) → inactive (grey).
```
1. Verify Phone   2. Destination   3. Travel Date
4. Select Flight  5. Passenger     6. Contact/Payment
7. Confirmed
```

### TTS / STT
- Web Speech Synthesis (TTS) reads every agent response aloud (toggleable)
- Web Speech Recognition (STT) lets users speak instead of type (mic button)

---

## 14. Environment Variables (`.env`)

```env
ANTHROPIC_API_KEY=...       # Required — Claude Haiku
EXTERNAL_API_BASE=...       # AWS flight API (pre-filled)
DATABASE_PATH=./airline.db  # SQLite file path

# Optional — real SMS
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=...

# Optional — real email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASS=...
SMTP_FROM=...
```

Without Twilio: OTP and booking confirmations print to the server console.
Without SMTP: PDF ticket is logged to console, not emailed.

---

## 15. Running the Project

```bash
pip install -r requirements.txt
# create .env from .env.example and add ANTHROPIC_API_KEY
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open browser: `http://localhost:8000`

The frontend is served by FastAPI at `/app/index.html` — there is no separate frontend server.
