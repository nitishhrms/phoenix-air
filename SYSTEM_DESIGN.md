# Phoenix Air — Voice AI Booking Agent: System Design

---

## 1. What This System Is

A voice-first airline booking agent that takes a caller from "hello" to a confirmed, paid booking through natural conversation. It runs on FastAPI + LangGraph and uses a cascade of Claude Haiku, Groq (Llama 3.1 8B), embeddings, and rule-based logic to be both fast and safe.

---

## 2. High-Level Architecture

```
User (voice / chat)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  FastAPI  POST /api/voice                                     │
│  - Load session state from SQLite                             │
│  - Inject user_input into state                               │
│  - Run LangGraph agent (15s timeout, 1 retry)                 │
│  - Save session state                                         │
│  - Return VoiceResponse (response, conv_state, flights, …)    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph                                         │
│                                                               │
│  security_guard                                               │
│       │ (blocked?) → end_node                                 │
│       ▼                                                       │
│  input_classifier  →  answer_validator ─────┐                 │
│       │ (query/question)                    │ (invalid/OOD)   │
│       ▼                                     ▼                 │
│  intent_router ──────────────────────────► END               │
│       │                                                       │
│       ├─ policy/general_qa  → rag_policy                      │
│       ├─ greeting           → greeting                        │
│       ├─ transfer           → transfer                        │
│       ├─ out_of_domain      → out_of_domain                   │
│       ├─ field_correction   → field_correction                │
│       ├─ customer_contact   → customer_contact                │
│       └─ normal (route by conv_state):                        │
│             SELECTING_LANGUAGE    → language_selection        │
│             COLLECTING_DEPARTURE  → resolve_airport           │
│             COLLECTING_DESTINATION→ resolve_airport           │
│             COLLECTING_DATE       → search_flights            │
│             PRESENTING_FLIGHTS    → present_flights           │
│             COLLECTING_PASSENGER  → collect_passenger         │
│             COLLECTING_CONTACT    → book_flight               │
│             COLLECTING_PAYMENT    → payment                   │
│             POST_BOOKING          → greeting                  │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Conversation State Machine

```
IDLE
  │
  ▼
SELECTING_LANGUAGE  ──────────────────────────────────┐
  │                                                   │
  ▼                                                   │
COLLECTING_PHONE                                      │
  │                                                   │
  ▼                                                   │
VERIFYING_OTP                                         │  field_correction
  │                                                   │  can jump BACK
  ▼                                                   │  to any earlier
COLLECTING_DEPARTURE ◄────────────────────────────────┤  state and clear
  │                                                   │  downstream fields
  ▼                                                   │
COLLECTING_DESTINATION ◄──────────────────────────────┤
  │                                                   │
  ▼                                                   │
COLLECTING_DATE ◄─────────────────────────────────────┤
  │                                                   │
  ▼                                                   │
PRESENTING_FLIGHTS                                    │
  │                                                   │
  ▼                                                   │
COLLECTING_PASSENGER ◄────────────────────────────────┤
  │                                                   │
  ▼                                                   │
COLLECTING_CONTACT ◄──────────────────────────────────┘
  │
  ▼
COLLECTING_PAYMENT
  │
  ▼
POST_BOOKING  →  DONE
```

**Downstream clearing rule:** correcting an earlier field wipes everything after it.
Example: user corrects `departure` while at `COLLECTING_CONTACT` → clears destination, date, flights, passenger, contact, payment.

---

## 4. INPUT GUARDS

Input guards sit at the very front of the pipeline and decide whether to process input at all, and what *type* of input it is.

### 4.1 Security Guard (`nodes/security_guard.py`)

**Where:** First node. Runs on every single request before anything else.

**What it blocks:**
- Prompt injection (`ignore previous instructions`, `you are now DAN`)
- SQL injection patterns
- XSS payloads
- System probes (`what is your system prompt`, `list your instructions`)

**How:** Rule-based pattern matching in `services/security_guard.py`. No LLM involved — pure regex/substring checks for speed and reliability.

**Output:** Sets `blocked=True` → graph routes to `end_node` → returns a safe hardcoded response. User never reaches any LLM.

```
User input
    │
    ▼
[security_guard] ──blocked──► end_node → "I can't process that request"
    │
  safe
    │
    ▼
[input_classifier]
```

### 4.2 Input Classifier (`nodes/input_classifier.py`)

**Where:** Second node. Runs after security guard passes.

**What it does:** Classifies input into one of three types before any domain logic runs:

| Type | Meaning | Example |
|------|---------|---------|
| `answer` | User is responding to bot's question | "New York", "yes", "john@email.com" |
| `question` | User is asking for information | "What's the refund policy?" |
| `query` | User is making a request | "Book a flight", "Start over" |

**3-Layer Pipeline (fastest first):**

```
Layer 0 — Hardcoded shortcuts (no LLM, ~0ms)
  ├─ Auth states (SELECTING_LANGUAGE, COLLECTING_PHONE, VERIFYING_OTP) → always "answer"
  ├─ Single digit ("1"–"5") at PRESENTING_FLIGHTS → always "answer"
  └─ _QUERY_SHORTCUTS keywords ("update my", "start over", "hello") → always "query"

Layer 1 — Groq Llama 3.1 8B (~100ms, free tier)
  └─ Single-word classification: answer / question / query

Layer 2 — Claude Haiku fallback (~600ms)
  └─ Same classification, used only if Groq is unavailable or errors
```

**Prompting technique used:**
- Temperature = 0 (deterministic)
- System prompt includes current `conv_state` and a hint about what the bot just asked
- `max_tokens=5` — forces single-word response, prevents rambling

### 4.3 Answer Validator (`nodes/answer_validator.py`)

**Where:** Runs when `input_type == "answer"`. Acts as a global redirect layer.

**Two jobs:**

**Job 1 — Global Intent Override (catch misclassified inputs)**
Even if the classifier said "answer", the validator checks whether the input is actually a question or query. This catches cases like user saying "what's the refund policy?" during phone collection — the classifier might label it "answer" since auth states always return "answer", but the validator overrides it.

```
_UNAMBIGUOUS_QUERY_KWS shortcuts (no LLM, ~0ms)
    ↓ (if not matched)
Groq / Haiku: "Is this a question/query or an answer to the bot?"
    ↓ (if question/query)
Re-route to intent_router instead of booking node
```

**Job 2 — Format Validation per booking state:**

```
State               L1 Regex                      L2 LLM correction
──────────────────────────────────────────────────────────────────
DEPARTURE/DEST      letters + spaces + hyphens     "Did you mean Los Angeles?"
DATE                7 date format patterns          "Did you mean Aug 15, 2026?"
PASSENGER           2+ words                       "Did you mean John Smith?"
CONTACT             email regex OR phone regex      "Did you mean +1 408..."
PRESENTING_FLIGHTS  single digit 1-5               (hardcoded hint)
```

**Pending correction flow** (spelling correction UX):
```
Turn 1: User says "Los Angles"
        → L2 LLM returns {"corrected": "Los Angeles", "confident": true}
        → Bot says: "Did you mean Los Angeles?"
        → Stores "Los Angeles" in pending_correction

Turn 2a: User says "yes"
         → Uses pending_correction value, continues booking ✓

Turn 2b: User says "no"
         → Clears pending_correction, asks again

Turn 2b: User says "San Francisco"
         → Clears pending_correction, validates new input
```

---

## 5. HALLUCINATION GUARDS

Multiple layers prevent the LLM from making up flight numbers, prices, cities, confirmation codes, or policies.

### 5.1 LLM Hallucination Guard — Core (`services/llm.py → _hallucination_check()`)

**Where:** Runs on EVERY `chat_response()` call before the response is returned to the user.

**How it works:**
```python
# After LLM generates a response:
system = """You are a strict fact-checker for an airline chatbot.
Given a booking context (what data the bot has), user input, and bot response:
Reply EXACTLY 'PASS' if the response uses only facts present in context.
Reply EXACTLY 'FAIL' if the response invents flights, prices, cities, dates,
or confirmation numbers not in context."""

# Inputs provided:
context_str  = JSON.dumps(booking_context_dict)
user_input   = current user message
response     = LLM-generated response candidate

# Result:
PASS → return response to user
FAIL → return None → node uses hardcoded fallback instead
```

**5s timeout** on the hallucination check itself. If the check times out, it defaults to PASS (favour availability over strict validation).

**Explicit system prompt rules (also in the main prompt):**
```
"Never invent flight numbers, prices, confirmation codes, airports, or dates.
Use ONLY data from the booking context block below.
If a fact isn't in the context, don't say it."
```

### 5.2 RAG Policy Hallucination Guard (`nodes/rag_policy.py`)

**Where:** Runs after TF-IDF retrieval for policy questions.

**How:** After generating an answer grounded in a retrieved policy document, a second LLM call checks:
> "Does this response contradict or exceed the retrieved policy document?"

If the check fails, the node falls through to the next retrieval layer (web search → LLM knowledge) rather than returning a hallucinated policy answer.

**When it's skipped:**
- Web search results → trusted external source, guard skipped
- General QA docs → lower stakes, guard skipped
- Pure LLM knowledge → standard timeout guard applies

### 5.3 Self-Reflection Guard (`services/llm.py → self_reflect_confirmation()`)

**Where:** `nodes/present_flights.py` — after the bot confirms a flight selection.

**How:** After generating "Great choice! Delta DL123...", a second Haiku call checks:
> "Does this response correctly name the flight the user selected?"

```python
PASS → return response
FAIL → regenerate with explicit instruction:
       "Confirm EXPLICITLY that user selected {airline} flight {number}.
       Then ask for their full name."
```

This prevents the bot from confirming the wrong flight (e.g., confusing DL123 with UA456 when multiple flights were presented).

### 5.4 Timeout Guard (`services/llm.py`)

**Where:** Every LLM call.

**How:** 8-second timeout using `concurrent.futures.ThreadPoolExecutor`. If LLM doesn't respond in 8s:
- Returns `None`
- Node uses its hardcoded fallback string
- User never waits indefinitely

### 5.5 Chain-of-Thought Flight Analysis (reduces hallucination by grounding)

**Where:** `nodes/search_flights.py`

**How:** Instead of asking Haiku to just "pick the best flight", it's forced to reason step by step:

```
Prompt structure:
"Analyze these flights step by step. Think through price, duration, stops, and value.
Then format your response EXACTLY as:
THINKING: <your reasoning>
BEST_VALUE: <flight number>
CHEAPEST: <flight number>
FASTEST: <flight number>
SUMMARY: <one sentence for the user>"
```

The structured output is then parsed to tag each flight with `best_value`, `cheapest`, `fastest` flags. The THINKING section forces the LLM to commit to logic before making a claim — dramatically reducing "I'll just say this flight is best" fabrication.

---

## 6. PROMPTING TECHNIQUES

### 6.1 System Prompt Caching (`services/llm.py`)

The static system rules are sent with `cache_control: {"type": "ephemeral"}`. Anthropic caches this block server-side for 5 minutes. Subsequent calls in the same window skip re-encoding the system prompt, saving ~200ms per call.

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": STATIC_SYSTEM_PROMPT,  # ← cached
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": dynamic_context_block,  # ← NOT cached (changes each turn)
            },
            {
                "type": "text",
                "text": user_input,
            },
        ],
    }
]
```

### 6.2 Chain-of-Thought (CoT)

Used in flight analysis (`search_flights.py`) and RAG grounding. Forces the LLM to show its reasoning before making a claim. Reduces hallucination and improves accuracy on multi-step decisions.

### 6.3 Few-Shot via Intent Anchors (`nodes/intent_router.py`)

Rather than relying solely on the LLM to classify intent, the intent router pre-embeds 7–9 example sentences per intent using `all-MiniLM-L6-v2`:

```python
INTENT_ANCHORS = {
    "transfer":    ["speak to a human", "talk to an agent", "connect me to support", ...],
    "policy":      ["what is the refund policy", "how much does an extra bag cost", ...],
    "greeting":    ["hello there", "hi good morning", "hey how are you", ...],
    "out_of_domain": ["what is the capital of France", "tell me a joke", ...],
    ...
}
```

At runtime, the user's input is embedded and compared against each intent's mean vector. This is effectively few-shot classification without LLM tokens — fast (~5ms), free, and deterministic.

### 6.4 Structured Output Parsing

Used in several places to make LLM output predictable:

| Location | Format expected |
|----------|----------------|
| Flight analysis (CoT) | `THINKING: ... BEST_VALUE: ... CHEAPEST: ...` |
| Answer validator L2 | `{"corrected": "...", "confident": true/false}` |
| City validator | `{"has_airport": true/false, "corrected": "..."}` |
| LLM Judge | `RELEVANCE:N ACCURACY:N HELPFULNESS:N SAFETY:pass REASON:...` |

Structured formats eliminate ambiguity and make parsing reliable without a full JSON schema.

### 6.5 Temperature = 0

All classification tasks (input_classifier, intent_router, answer_validator, hallucination guard) use `temperature=0`. This makes outputs deterministic and reproducible — critical when routing wrong sends the user down the wrong path.

### 6.6 Context Grounding (`services/llm.py`)

Every `chat_response()` call includes a dynamic booking context block:

```
[BOOKING CONTEXT]
departure_city: New York (JFK)
destination_city: Los Angeles (LAX)
travel_date: 2026-06-15
passenger_first: John
passenger_last: Smith
selected_flight: Delta DL456, $245.50, departs 08:30
...
```

The LLM is explicitly told to use only this data. Combined with the hallucination guard, this prevents fabrication.

### 6.7 Explicit Negations in System Prompt

```
"Never invent flight numbers, prices, confirmation codes, airports, or dates.
Never pretend to have access to the internet or real-time data.
Never make up policy details — only state what's in the retrieved document."
```

Explicit negations outperform implicit constraints ("only use provided data") in reducing hallucination rate.

### 6.8 Prompt Caching on Field Correction (`nodes/field_correction.py`)

The system instruction for field detection is cached with `cache_control: ephemeral`:
```python
system=[{
    "type": "text",
    "text": "The user is booking a flight and wants to correct something...",
    "cache_control": {"type": "ephemeral"},
}]
```

---

## 7. NODE REFERENCE

| Node | File | Role | LLM Used |
|------|------|------|----------|
| security_guard | nodes/security_guard.py | Block injection/probes | ❌ Rule-based |
| input_classifier | nodes/input_classifier.py | answer/question/query | Groq → Haiku |
| answer_validator | nodes/answer_validator.py | Validate + redirect answers | Groq → Haiku |
| intent_router | nodes/intent_router.py | Detect intent (3-layer) | Embeddings + Haiku |
| language_selection | nodes/language_selection.py | Pick language | ❌ Rule-based |
| resolve_airport | nodes/resolve_airport.py | City → IATA (4-layer) | Haiku (L4 only) |
| search_flights | nodes/search_flights.py | Date parse + flight search + CoT | Haiku (CoT) |
| present_flights | nodes/present_flights.py | Flight selection + sort | Haiku |
| collect_passenger | nodes/collect_passenger.py | Parse passenger name | Haiku |
| book_flight | nodes/book_flight.py | Contact + API booking | Haiku |
| payment | nodes/payment.py | Mock payment + PDF | Haiku |
| rag_policy | nodes/rag_policy.py | Policy + general QA | Haiku (grounded) |
| greeting | nodes/greeting.py | Greet / post-booking | Haiku |
| field_correction | nodes/field_correction.py | Mid-booking corrections | Haiku (L2) |
| out_of_domain | nodes/out_of_domain.py | Reject off-topic | ❌ Hardcoded |
| transfer | nodes/transfer.py | Connect to human | Haiku |
| customer_contact | nodes/customer_contact.py | Complaints / issues | Haiku |
| auth | nodes/auth.py | Phone + OTP auth | ❌ Rule-based |

---

## 8. RAG PIPELINE (`nodes/rag_policy.py`)

```
User question
      │
      ├─ POLICY PATH (refund, baggage, cancellation, etc.)
      │       │
      │       ▼
      │   L1: Rule-based hardcoded answers (~0ms)
      │       │ (not found)
      │       ▼
      │   L2: TF-IDF retrieval over 14 policy documents
      │       │ (found)
      │       ▼
      │   Haiku grounded to retrieved doc
      │       │
      │       ▼
      │   Hallucination guard → PASS/FAIL
      │       │ (FAIL → fall through)
      │
      └─ GENERAL QA PATH (wifi, meals, aircraft, pets, safety, etc.)
              │
              ▼
          Airline relatedness check (50+ keyword list)
              │ (not airline-related)
              ▼
          Hardcoded OOD response
              │ (airline-related)
              ▼
          L1: Rule-based hardcoded general answers
              │ (not found)
              ▼
          L2: TF-IDF retrieval over general docs
              │ (found)
              ▼
          Haiku grounded to doc (no hallucination guard — trusted source)
              │ (not found)
              ▼
          L3: DuckDuckGo web search
              │ (results found)
              ▼
          Haiku grounded to search results
              │ (no results / timeout)
              ▼
          L4: Pure Haiku knowledge (no retrieval)
              │ (timeout / error)
              ▼
          L5: Escalate → customer_contact node
```

---

## 9. SERVICES LAYER

### `services/llm.py` — Central LLM Gateway

All LLM calls for response generation go through `chat_response()`. It provides:
- Prompt caching (system prompt)
- Context grounding (booking state injection)
- 8s timeout guard
- Hallucination check on every response
- Language instruction injection (8 languages)
- Returns `None` on failure → caller uses hardcoded fallback

### `services/llm_judge.py` — Background Observability

After every user turn, a background thread runs an LLM evaluation:

```
Scores (1–5):  RELEVANCE · ACCURACY · HELPFULNESS
Flag (pass/fail): SAFETY
```

Results saved to SQLite. Accessible at `GET /api/observability`. Never blocks the user — purely async.

### `services/external_api.py` — Flight Search & Booking

- `search_flights(src, dst, date)` → `{flights: [...]}`
- `search_flights_multi(src, dst, dates)` → parallel search over multiple dates via `ThreadPoolExecutor`
- `book_flight(src, dst, date, flight_id, first_name, last_name)` → `{confirmationNumber: ...}`
- Error codes: `NO_FLIGHTS`, `INVALID_DATE`, `API_ERROR`

### `services/payment.py` — Mock Payment

`mock_charge(amount, contact, confirmation)` always succeeds. Logs to SQLite `payments` table. Returns `{transaction_id: "TXN-XXXXXXXX"}`.

### `services/pdf_ticket.py` — Boarding Pass PDF

Generates a boarding pass-style PDF using `reportlab`. Falls back to plain UTF-8 text if reportlab is not installed. Sent via email (SMTP) or logged to console for phone contacts.

---

## 10. DATA FLOW — FULL EXAMPLE (New York → Los Angeles booking)

```
Turn 1: User: "hello"
  security_guard: PASS
  input_classifier: auth state → answer
  answer_validator: "hello" → _UNAMBIGUOUS_QUERY_KWS → reroute as greeting
  intent_router: intent=greeting
  greeting: "Welcome to Phoenix Air! Select your language"
  → conv_state: SELECTING_LANGUAGE

Turn 2: User clicks "English" card (sends "1")
  input_classifier: auth state → answer
  language_selection: "1" → en confirmed
  → conv_state: COLLECTING_PHONE

Turn 3: User: "+14085550001"
  input_classifier: auth state → answer
  auth: phone valid → OTP sent
  → conv_state: VERIFYING_OTP

Turn 4: User: "374115"
  input_classifier: auth state → answer
  auth: OTP verified
  → conv_state: COLLECTING_DEPARTURE

Turn 5: User: "New York"
  input_classifier: Groq → answer
  answer_validator: _CITY_RE matches → valid
  intent_router: intent=normal, conv_state=COLLECTING_DEPARTURE
  resolve_airport: L1 DB match → JFK
  → departure_city="New York", departure_iata="JFK", conv_state: COLLECTING_DESTINATION

Turn 6: User: "Los Angles" (typo)
  input_classifier: Groq → answer
  answer_validator: _CITY_RE matches (letters+spaces) → valid format
  resolve_airport: L1 miss → L2 fuzzy → L3 embedding → suggests "Los Angeles"
  → "Did you mean Los Angeles (LAX)?"

Turn 7: User: "yes"
  resolve_airport: pending_correction → LAX confirmed
  → destination_city="Los Angeles", destination_iata="LAX", conv_state: COLLECTING_DATE

Turn 8: User: "August 15"
  search_flights: regex matches → 2026-08-15
  external_api: GET flights JFK→LAX 2026-08-15 → 3 flights returned
  CoT analysis: THINKING... BEST_VALUE: UA456 CHEAPEST: DL123 FASTEST: AA789
  → conv_state: PRESENTING_FLIGHTS, flights: [...]

Turn 9: User clicks "Select" on Delta DL123 (sends "1", displays "Delta DL123 — $189.00")
  present_flights: _parse_selection("1") → idx=0 → selected_flight=DL123
  self_reflect: "Does response confirm Delta DL123?" → PASS
  → conv_state: COLLECTING_PASSENGER

Turn 10: User: "John Smith"
  collect_passenger: split → first="John", last="Smith"
  → conv_state: COLLECTING_CONTACT

Turn 11: User: "john@email.com"
  book_flight: email regex matches → contact_type=email
  external_api: POST book flight → confirmationNumber="PHN-X7K2M9"
  save_booking: SQLite insert
  → conv_state: COLLECTING_PAYMENT, confirmation="PHN-X7K2M9"

Turn 12: UI shows payment card — User clicks "Confirm & Pay $189.00" (sends "yes")
  payment: _YES matches → mock_charge($189.00) → txn_id="TXN-A1B2C3D4"
  generate_ticket: PDF bytes
  send_ticket: logged (SMTP not configured)
  LLM response: "Payment of $189.00 confirmed! Confirmation: PHN-X7K2M9..."
  Hallucination guard: context has all facts → PASS
  → conv_state: POST_BOOKING

  UI: addConfirmCard("PHN-X7K2M9") shown with download button

Turn 13: User clicks "Download Ticket PDF"
  GET /api/ticket/PHN-X7K2M9
  get_booking_by_confirmation() → booking dict
  generate_ticket() → PDF
  → PDF download (200 OK, application/pdf)
```

---

## 11. WHERE EACH CONCEPT LIVES — QUICK REFERENCE

### Input Guards
| Guard | Location | Technique |
|-------|----------|-----------|
| Injection / XSS / SQL block | `nodes/security_guard.py` + `services/security_guard.py` | Regex / substring |
| Input type classification | `nodes/input_classifier.py` | Shortcuts + Groq + Haiku |
| Answer format validation | `nodes/answer_validator.py` | Regex per state + LLM correction |
| Auth input redirect | `nodes/auth.py` | `_looks_like_question()` hardcoded |

### Hallucination Guards
| Guard | Location | Technique |
|-------|----------|-----------|
| Response vs context check | `services/llm.py → _hallucination_check()` | LLM judge (PASS/FAIL) |
| Policy answer vs document | `nodes/rag_policy.py` | `check_response()` |
| Flight confirmation check | `nodes/present_flights.py` | `self_reflect_confirmation()` |
| LLM timeout fallback | `services/llm.py` | 8s ThreadPoolExecutor |
| Explicit negations in prompt | `services/llm.py → STATIC_SYSTEM_PROMPT` | Prompt engineering |

### Prompting Techniques
| Technique | Location | Purpose |
|-----------|----------|---------|
| Prompt caching | `services/llm.py`, `nodes/field_correction.py` | Speed / cost |
| Chain-of-Thought | `nodes/search_flights.py`, `services/llm.py` | Flight analysis accuracy |
| Few-shot (intent anchors) | `nodes/intent_router.py` | Embedding-based classification |
| Structured output | Multiple nodes | Reliable parsing |
| Temperature = 0 | All classifiers | Deterministic routing |
| Context grounding | `services/llm.py` | Prevent fabrication |
| Explicit negations | `services/llm.py` | Reduce hallucination |
| Self-reflection | `services/llm.py`, `nodes/present_flights.py` | Verify LLM claims |

---

## 12. TECH STACK

| Component | Technology |
|-----------|-----------|
| API server | FastAPI (Python) |
| Agent orchestration | LangGraph (StateGraph) |
| Primary LLM | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Fast classifier | Groq Llama 3.1 8B Instant |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| TF-IDF retrieval | scikit-learn TfidfVectorizer |
| Web search | DuckDuckGo (via `duckduckgo_search`) |
| Database | SQLite (sessions, bookings, payments, users, judge_logs) |
| PDF generation | ReportLab |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| Voice (TTS/STT) | Web Speech API (browser-native) |
