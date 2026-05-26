# Phonely Airline — Architecture & Prompting Techniques

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Folder Structure](#3-folder-structure)
4. [Data Model — AirlineState](#4-data-model--airlinestate)
5. [Conversation States (State Machine)](#5-conversation-states-state-machine)
6. [LangGraph — How the Graph Works](#6-langgraph--how-the-graph-works)
7. [Every Node Explained](#7-every-node-explained)
8. [Every Service Explained](#8-every-service-explained)
9. [Database Schema](#9-database-schema)
10. [API Endpoints](#10-api-endpoints)
11. [Prompting Techniques](#11-prompting-techniques)
12. [Feature Deep Dives](#12-feature-deep-dives)
13. [Request Lifecycle — End to End](#13-request-lifecycle--end-to-end)

---

## 1. Project Overview

Phonely Airline is an AI-powered voice booking assistant for "Phoenix Air". A user can:
- Call in (simulated via browser mic) or use the web chat UI
- Book a flight in 8 languages
- Get policy answers (refunds, baggage, etc.)
- Download a PDF ticket after booking
- Log in / sign up with a web account

The backend is a Python FastAPI server. Every user message goes through a LangGraph state machine. Claude Haiku (Anthropic's fastest/cheapest model) powers all natural language understanding and response generation.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI (Python) |
| AI orchestration | LangGraph (StateGraph) |
| LLM | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Database | SQLite (via `sqlite3` stdlib) |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| PDF generation | ReportLab |
| OTP / SMS | Twilio (dev mode: prints to console) |
| Airport search | TF-IDF cosine similarity (scikit-learn) |
| Date parsing | dateutil + regex + LLM fallback |
| External flights | Custom external flight search API |

---

## 3. Folder Structure

```
phonely-airline/
│
├── main.py                  # FastAPI app, all HTTP endpoints, background judge
├── graph.py                 # Builds the LangGraph StateGraph
├── state.py                 # AirlineState TypedDict + empty_state()
│
├── nodes/                   # One file per conversation node
│   ├── security_guard.py    # Blocks attacks before anything runs
│   ├── intent_router.py     # Classifies user intent each turn
│   ├── language_selection.py# Language menu (new)
│   ├── resolve_airport.py   # Collects departure + destination
│   ├── search_flights.py    # Parses date, fetches + ranks flights, CoT
│   ├── present_flights.py   # Handles flight selection + sort requests
│   ├── collect_passenger.py # Collects passenger name
│   ├── book_flight.py       # Confirms booking, saves to DB
│   ├── payment.py           # Handles payment collection
│   ├── rag_policy.py        # Answers policy questions
│   ├── transfer.py          # Hands off to human agent
│   ├── greeting.py          # Handles greetings
│   ├── out_of_domain.py     # Handles off-topic requests
│   ├── customer_contact.py  # Collects contact info
│   ├── field_correction.py  # Lets user correct a previously entered field
│   └── auth.py              # Phone OTP authentication
│
├── services/
│   ├── llm.py               # Central LLM caller (all prompting + guards)
│   ├── llm_judge.py         # LLM as Judge for observability (new)
│   ├── auth.py              # OTP + web account login/signup
│   ├── rag.py               # Policy retrieval (TF-IDF + Haiku answer)
│   ├── security_guard.py    # 3-layer input validation
│   ├── external_api.py      # Flight search API client
│   ├── pdf_ticket.py        # PDF boarding pass generator
│   └── timeout_guard.py     # Wraps graph.invoke() with timeout + retry
│
├── db/
│   ├── database.py          # All SQLite helpers + init_db()
│   └── seed_airports.py     # Seeds airport data on first run
│
└── frontend/
    └── index.html           # Complete single-page UI
```

---

## 4. Data Model — AirlineState

`state.py` defines the single object that flows through the entire graph on every turn.

```python
class AirlineState(TypedDict):
    # Identity
    session_id:        str        # UUID per conversation
    caller_phone:      str        # phone number (voice calls)
    user_id:           int | None # web login account ID

    # Language
    language:          str        # en | es | fr | hi | zh | ar | pt | de

    # Authentication
    authenticated:     bool
    auth_phone:        str | None

    # Conversation
    conv_state:        str        # which step we're in (see state machine below)
    messages:          list       # LangChain message history
    user_input:        str        # what the user just said
    response:          str        # what the AI will say back

    # Booking data (filled step by step)
    departure_city:    str | None
    departure_iata:    str | None
    destination_city:  str | None
    destination_iata:  str | None
    travel_date:       str | None     # YYYY-MM-DD
    travel_dates:      list | None    # multiple dates (e.g. "Aug 5 or Aug 6")
    flights:           list | None    # raw flight objects from API
    selected_flight:   dict | None
    passenger_first:   str | None
    passenger_last:    str | None
    contact:           str | None     # email or phone for ticket
    contact_type:      str | None     # "phone" | "email"
    confirmation:      str | None     # PHN-XXXXXX

    # Payment
    flight_price:      float | None
    payment_confirmed: bool
    transaction_id:    str | None

    # Routing / control
    intent:            str            # normal | transfer | policy | greeting | ...
    retry_count:       int
    end_call:          bool
    transfer:          bool
    blocked:           bool           # set by security_guard
    flight_sort:       str | None     # "price" | "time"
    cot_analysis:      dict | None    # CoT flight analysis result
```

**Key rule:** Every node receives the full state dict and returns a partial dict with only the fields it changed. LangGraph merges the returned dict into the state. Nothing is ever lost.

---

## 5. Conversation States (State Machine)

Each turn, `conv_state` tells the graph which node to route to next.

```
SELECTING_LANGUAGE        → language_selection_node
       ↓ (language chosen)
COLLECTING_PHONE          → auth_node (sends OTP)
       ↓ (OTP sent)
VERIFYING_OTP             → auth_node (checks code)
       ↓ (verified)
COLLECTING_DEPARTURE      → resolve_airport_node
       ↓
COLLECTING_DESTINATION    → resolve_airport_node
       ↓
COLLECTING_DATE           → search_flights_node
       ↓
PRESENTING_FLIGHTS        → present_flights_node
       ↓ (user picks a flight)
COLLECTING_PASSENGER      → collect_passenger_node
       ↓
COLLECTING_CONTACT        → book_flight_node
       ↓
COLLECTING_PAYMENT        → payment_node
       ↓
BOOKING_CONFIRMED         → (session cleaned up)
DONE                      → end_node (call ends)
```

**Layer 1 intents** override this flow at any point:
- User says "transfer me" → `transfer_node` (any state)
- User asks "what's your refund policy?" → `rag_policy_node` (any state)
- User says "hello" → `greeting_node`
- User tries to change a field → `field_correction_node`

---

## 6. LangGraph — How the Graph Works

LangGraph is a library for building stateful agent graphs. Think of it as a flowchart where each box is a Python function.

```python
# graph.py (simplified)
graph = StateGraph(AirlineState)

graph.set_entry_point("security_guard")

graph.add_conditional_edges(
    "security_guard",
    after_security,             # returns "end_node" or "intent_router"
    {...}
)

graph.add_conditional_edges(
    "intent_router",
    should_continue,            # reads intent + conv_state → returns node name
    {...}
)

# All other nodes go straight to END after running
for node in ["language_selection", "resolve_airport", ...]:
    graph.add_edge(node, END)
```

**One turn = one full graph traversal:**

```
security_guard → intent_router → [exactly one target node] → END
```

The graph does NOT loop internally. Each HTTP request triggers one traversal. The conversation continues because the state is saved to SQLite and reloaded on the next request.

**`should_continue()` — the routing brain:**
```python
def should_continue(state):
    intent = state.get("intent", "normal")
    conv_state = state.get("conv_state", "IDLE")

    # Layer 1: intent overrides (interrupt anything)
    if intent == "transfer": return "transfer"
    if intent == "policy":   return "rag_policy"
    if intent == "greeting": return "greeting"
    # ... etc

    # Layer 2: state-driven routing
    routing = {
        "SELECTING_LANGUAGE":   "language_selection",
        "COLLECTING_DEPARTURE": "resolve_airport",
        "COLLECTING_DATE":      "search_flights",
        "PRESENTING_FLIGHTS":   "present_flights",
        # ... etc
    }
    return routing.get(conv_state, "end_node")
```

---

## 7. Every Node Explained

### `security_guard_node`
**Runs first, every turn.** Validates the raw user input before any LLM sees it.
- Layer 1: length limit (500 chars)
- Layer 2: regex for prompt injection, SQL injection, XSS, system probes
- Layer 3: Haiku classifier for subtle attacks not caught by regex
- On block: sets `state["blocked"] = True` → graph routes to `end_node`

### `intent_router_node`
**Runs second, every turn.** Asks Haiku: "What is the user trying to do?"
- Possible intents: `normal`, `transfer`, `policy`, `restart`, `out_of_domain`, `greeting`, `field_correction`, `customer_contact`
- Bypassed (returns `normal`) for: `SELECTING_LANGUAGE`, `COLLECTING_PHONE`, `VERIFYING_OTP` — because in those states we already know what to do

### `language_selection_node` *(new)*
- Turn 1 (no input): displays 8-language menu
- Turn 2+: detects language via keyword map → LLM fallback → sets `state["language"]`
- Advances to `COLLECTING_PHONE`

### `resolve_airport_node`
- Collects departure and destination cities one at a time
- Uses `database.resolve_airport()` (exact match + alias match)
- Uses `database.suggest_airports()` (fuzzy match) for suggestions
- IATA code + city name stored in state

### `search_flights_node`
- Parses travel date: regex → dateutil → Haiku (3-layer)
- Supports multi-date queries: "August 5 or August 6" → searches both dates
- Calls external flight API
- Runs **CoT analysis** on results (see Feature Deep Dives)
- Tags each flight: `best_value`, `cheapest`, `fastest`

### `present_flights_node`
- **Sort detection first:** "cheapest" / "fastest" → re-sorts + returns new list
- **Flight selection:** by number, ordinal word, airline name
- **Self-reflection:** verifies generated response mentions the correct flight
- Advances to `COLLECTING_PASSENGER`

### `collect_passenger_node`
Collects passenger full name. Claude extracts first/last from natural input ("my name is John Smith").

### `book_flight_node`
Collects contact info (email or phone for ticket). Generates confirmation number (PHN-XXXXXX). Saves booking to SQLite. Upserts user profile for returning customer detection.

### `payment_node`
Collects payment details. In dev mode, simulates a successful payment. Saves transaction to `payments` table.

### `auth_node`
- `COLLECTING_PHONE`: asks for phone, generates + sends OTP via Twilio (or dev console)
- `VERIFYING_OTP`: checks OTP code, marks `authenticated = True`

### `rag_policy_node`
Answers policy questions using **Retrieval Augmented Generation** (see Prompting Techniques).

### `transfer_node`
Tells the user they're being transferred. Sets `transfer = True` in state.

### `greeting_node`
Handles "hello", "hi", etc. with a friendly Claude-generated greeting.

### `out_of_domain_node`
Handles off-topic requests ("tell me a joke") with a polite redirect.

### `customer_contact_node`
Collects or updates contact information mid-conversation.

### `field_correction_node`
Detects which field the user wants to change (departure city, date, etc.) and resets the state to re-collect it.

---

## 8. Every Service Explained

### `services/llm.py` — The Central LLM Hub

Everything that calls Claude Haiku goes through this file.

**`chat_response(task, context, user_input, max_tokens, language)`**
The main function every node uses to generate a response.
```
task       → what the AI needs to communicate this turn
context    → dict of booking data collected so far
user_input → what the user just said
language   → ISO code (en/es/fr/hi/zh/ar/pt/de)
```
Internally applies two guards automatically:
1. **8-second timeout** (via ThreadPoolExecutor)
2. **Hallucination guard** (second Haiku call verifies facts)

Returns `None` on failure → node uses its hardcoded fallback string.

**`cot_analyze_flights(flights, src, dst, date)`**
Chain-of-Thought flight analysis. Returns `{best_value_idx, cheapest_idx, fastest_idx, summary}`.

**`self_reflect_confirmation(selected, response)`**
Checks if the response correctly names the selected flight. Returns `True/False`.

### `services/llm_judge.py` — Observability *(new)*
Runs in a background thread after every response. Scores: RELEVANCE, ACCURACY, HELPFULNESS, SAFETY. Saves to `judge_logs` table.

### `services/auth.py`
OTP generation/verification + web account create/login. SHA-256 password hashing. Twilio SMS integration with dev-mode fallback.

### `services/rag.py`
Policy retrieval:
1. Keyword map (fast, accurate)
2. TF-IDF cosine similarity fallback
3. Haiku generates a conversational answer grounded in the retrieved policy text

### `services/security_guard.py`
Three-layer input guard (see Node section above).

### `services/timeout_guard.py`
Wraps the entire `graph.invoke()` call with a 15-second timeout and 1 retry. Prevents hung requests from blocking the server.

### `services/pdf_ticket.py`
Generates a PDF boarding pass from the booking record using ReportLab.

---

## 9. Database Schema

SQLite file at `./airline.db`.

```sql
-- Airport lookup table (seeded at startup)
airports (iata PK, city, name, aliases)

-- Active conversation sessions
sessions (session_id PK, conv_state, data JSON, updated_at)

-- Returning customer info
user_profiles (phone PK, first_name, last_name, last_from, last_to, call_count, updated_at)

-- Confirmed bookings
bookings (id, confirmation_number UNIQUE, session_id, flight_id, airline,
          flight_number, departure_time, arrival_time, src_iata, dst_iata,
          passenger_name, contact, contact_type, created_at)

-- OTP codes (TTL: 10 minutes)
otp_codes (phone PK, code, expires_at, verified)

-- Payment records
payments (id, transaction_id UNIQUE, confirmation_number, amount, contact,
          status, created_at)

-- Web login accounts  [NEW]
user_accounts (id, phone UNIQUE, email, first_name, last_name,
               password_hash, created_at)

-- LLM judge quality logs  [NEW]
judge_logs (id, session_id, node_name, user_input, response, language,
            relevance, accuracy, helpfulness, safety, reason, created_at)
```

---

## 10. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/voice` | Main webhook — processes one user turn |
| `POST` | `/api/auth/signup` | Create web account |
| `POST` | `/api/auth/login` | Login → returns user_id + name |
| `GET` | `/api/ticket/{confirmation}` | Download PDF ticket |
| `GET` | `/api/observability?limit=N` | Judge logs + aggregate scores |
| `GET` | `/health` | Health check |
| `GET` | `/` | Redirects to frontend |

---

## 11. Prompting Techniques

This is where the real AI engineering happens. Eight distinct prompting techniques are used across the system.

---

### Technique 1: Role Prompting

**What it is:** Give the model a persona and rules before anything else. The model stays "in character" for the entire conversation.

**Where:** `services/llm.py` → `_STATIC_SYSTEM`

```python
_STATIC_SYSTEM = (
    "You are Phoenix Air's friendly voice booking assistant.\n\n"
    "CORE RULES:\n"
    "- Reply in 1-2 short sentences only — this is a voice interface\n"
    "- Never invent flight numbers, prices, confirmation codes, airports, "
    "or dates — use ONLY data from the booking context block\n"
    "- Be warm and natural, not robotic or corporate\n"
    "- Stay focused on airline booking — politely decline anything off-topic\n"
    ...
)
```

**Why it works:** The model anchors its persona from the system prompt. Every subsequent instruction (task, context) is interpreted through this lens. Without it, the model might answer in paragraph form or make up prices.

---

### Technique 2: Prompt Caching

**What it is:** Mark the static portion of your prompt with `cache_control`. Anthropic stores it server-side so repeated calls don't re-process the same text. Saves cost (~90% cheaper on cached tokens) and reduces latency.

**Where:** Every LLM call — `llm.py`, `security_guard.py`, `rag.py`, `llm_judge.py`, `field_correction.py`

```python
system=[
    {
        "type": "text",
        "text": _STATIC_SYSTEM,           # same every call → cached
        "cache_control": {"type": "ephemeral"},
    },
    {
        "type": "text",
        "text": dynamic_system,            # changes every call → NOT cached
        # no cache_control here
    },
]
```

**The split:** Static rules go in block 1 (cached). Booking context + language instruction go in block 2 (dynamic, not cached). This way one cached copy covers all 8 languages — there's no need for 8 separate cached versions.

**Why it works:** The model re-uses its internal KV cache for the static block. You're only billed for the dynamic portion at full price.

---

### Technique 3: Chain-of-Thought (CoT) Prompting

**What it is:** Instead of asking the model to jump straight to an answer, instruct it to reason step by step first. The intermediate reasoning improves final answer quality.

**Where:** `services/llm.py` → `cot_analyze_flights()` → `_COT_SYSTEM`

```python
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
```

**Input sent to model:**
```
Route: New York → Los Angeles on 2026-08-15

Available flights:
Flight 1: United UA202, departs 6:00 AM, arrives 9:30 AM, duration 330min, $189.00, stops: 0
Flight 2: Delta DL445, departs 11:00 AM, arrives 2:45 PM, duration 345min, $159.00, stops: 1
Flight 3: American AA110, departs 3:00 PM, arrives 6:00 PM, duration 300min, $219.00, stops: 0

Analyze and recommend.
```

**Model's internal THINKING** is captured and then the structured labels (BEST_VALUE, CHEAPEST, FASTEST) are parsed as integers to tag the flights.

**Why it works:** Without CoT, the model picks the cheapest option by default. With it, it considers trade-offs — a flight 15 minutes shorter but $30 more expensive might be "best value" for a business traveler.

---

### Technique 4: Self-Reflection (Self-Critique)

**What it is:** After the model generates a response, a second model call checks whether that response is correct. If not, the response is regenerated with more explicit instructions.

**Where:** `services/llm.py` → `self_reflect_confirmation()`, `nodes/present_flights.py`

```python
# First: generate response
response = chat_response(task, ctx, user_input, language=language)

# Second: verify it
if not self_reflect_confirmation(selected, response):
    # Failed verification → regenerate with more explicit task
    task_retry = (
        f"Confirm EXPLICITLY that the user selected {selected['airline']} "
        f"flight {selected['flightNumber']}. Then ask for their full name."
    )
    response = chat_response(task_retry, ctx, user_input, language=language)
```

**The verifier prompt:**
```python
_REFLECT_SYSTEM = (
    "You are a quality checker for an airline booking AI. "
    "Check if the response correctly confirms the selected flight. "
    "Reply with EXACTLY 'PASS' if correct, 'FAIL' if the flight details are missing or wrong."
)
```

**Why it works:** Language models can occasionally "drift" — they understand the task but the generated text references a different flight number or airline. The self-reflection call catches this before the user sees it. It costs one extra Haiku call but eliminates a class of booking errors.

---

### Technique 5: Structured Output Prompting

**What it is:** Constrain the model's output to a specific format by describing the exact format in the system prompt, then parse the output programmatically.

**Used in three places:**

**CoT Analysis output:**
```
THINKING: ...reasoning...
BEST_VALUE: 2
CHEAPEST: 1
FASTEST: 3
SUMMARY: Flight 2 offers the best balance of price and convenience.
```
Parsed with: `line.startswith("BEST_VALUE:")` → extract integer.

**LLM Judge output:**
```
RELEVANCE:4 ACCURACY:5 HELPFULNESS:4 SAFETY:pass
REASON: Response was accurate but could have been more specific about the refund timeline.
```
Parsed token by token: `token.split(":")[1]`

**Security Classifier output:**
```
YES
```
or
```
NO
```
`max_tokens=8` forces the model to be concise. Checked with `answer.startswith("YES")`.

**Why it works:** Free-form model output is hard to use programmatically. By specifying exact format and enforcing it via the system prompt + `max_tokens`, you get reliable parseable output. The model follows format instructions extremely well when they are in the system prompt (not user message).

---

### Technique 6: Retrieval-Augmented Generation (RAG)

**What it is:** Instead of relying on the model's training knowledge (which may be wrong or outdated), retrieve the correct source document first, then ask the model to answer using ONLY that document.

**Where:** `services/rag.py` → `query_policy_llm()`

**Two-step process:**

Step 1 — Retrieval (keyword map + TF-IDF):
```python
# Keyword map first (fast, deterministic)
for keywords, policy_id in _KEYWORD_MAP:
    if any(kw in question.lower() for kw in keywords):
        return _docs[idx]   # exact match, no LLM needed

# TF-IDF cosine similarity fallback
q_vec = _vectorizer.transform([question])
scores = cosine_similarity(q_vec, _matrix).flatten()
return _docs[best_idx]
```

Step 2 — Grounded generation:
```python
system = (
    "Answer the passenger's question ONLY using the provided policy text. "
    "Do not add information not present in the policy text."
)
user_msg = f"Policy information:\n{context}\n\nPassenger question: {question}"
```

**Why it works:** Without RAG, the model might hallucinate a refund policy that doesn't exist. With RAG, it can only answer from the retrieved text. The `ONLY` instruction is critical — it grounds the model to the source document.

---

### Technique 7: Hallucination Guard (Fact-Checking Prompting)

**What it is:** After generating a response, a second model call checks whether the response invented any facts not present in the booking context.

**Where:** `services/llm.py` → `_hallucination_check()`

```python
_HAL_SYSTEM = (
    "You are a strict fact-checker for an airline booking assistant. "
    "Reply with exactly PASS if the response only uses facts from the context, "
    "or FAIL if it invents flights, prices, cities, dates, or confirmation numbers "
    "not present in the context."
)

# Sent to model:
f"Context:\n{context_str}\n\nUser said: {user_input}\nResponse: {response}"
```

If the verdict is `FAIL`, `chat_response()` returns `None` → the node uses its hardcoded fallback.

**Why it matters:** This is the most important safety guard for a booking system. A hallucinated confirmation number or wrong price creates real customer harm. The check costs 5 tokens to return `PASS` or `FAIL`.

**Optimization:** The check is skipped when `context_str` is empty (early in the conversation, before any booking data is collected — nothing to hallucinate about).

---

### Technique 8: Task Decomposition in Prompts

**What it is:** Instead of asking the model to figure out what to do, tell it exactly what the current task is. The `task` parameter in `chat_response()` is a short, specific instruction about what THIS response needs to accomplish.

**Where:** Every node that calls `chat_response()`

**Examples from the code:**

```python
# search_flights_node
task = f"Found {len(ranked)} flights from {src_city} to {dst_city} on {date_str}. "
       f"Tell the user and ask them to pick one from the cards. "
       f"CoT insight: {cot['summary']}"

# present_flights_node (sort)
task = f"User wants flights sorted by {sort_label}. Acknowledge and invite them to pick one."

# present_flights_node (selection)
task = (
    f"User selected {airline} flight {flight_number}. "
    "Confirm their selection enthusiastically and ask for their full name for the booking."
)

# auth_node (OTP sent)
task = f"OTP sent to {phone}. Tell user to check their SMS and enter the 6-digit code."
```

**Why it works:** The model doesn't need to figure out "what stage of booking are we in" — the node already did that reasoning. The task string tells it exactly what to say. The context block tells it the facts. The role prompt tells it how to say it. These three together produce consistent, accurate responses.

---

### Technique 9: LLM as Judge (Evaluation Prompting)

**What it is:** Use a separate model call to evaluate the quality of another model's output. This is called "LLM as Judge" and is a standard technique in production AI systems for observability.

**Where:** `services/llm_judge.py`

```python
_JUDGE_SYSTEM = (
    "You are an expert evaluator for an airline booking AI assistant. "
    "Evaluate the AI response on four dimensions:\n"
    "- RELEVANCE (1-5): Does the response directly address what the user said?\n"
    "- ACCURACY (1-5): Are all facts correct given the booking context?\n"
    "- HELPFULNESS (1-5): Does it move the conversation forward productively?\n"
    "- SAFETY (pass/fail): No hallucinated data, harmful content, or off-topic info?\n\n"
    "Reply in EXACTLY this format:\n"
    "RELEVANCE:N ACCURACY:N HELPFULNESS:N SAFETY:pass\n"
    "REASON:<one sentence explaining the lowest score>"
)
```

**When it runs:** In a **background daemon thread** after every API response. The user never waits for it.

**What it produces over time:**
```json
{
  "total": 200,
  "avg_relevance": 4.3,
  "avg_accuracy": 4.6,
  "avg_helpfulness": 4.1,
  "safety_pass_rate": 0.98,
  "logs": [...]
}
```

Visible at `GET /api/observability` and in the frontend's Observability modal.

**Why it's useful:** You can't manually review every AI response in production. The judge runs automatically and gives you aggregate quality metrics. If `avg_accuracy` drops, you know something changed in your prompts or context.

---

### Technique 10: Language Injection via Dynamic System Prompt

**What it is:** Add a language instruction to the dynamic (non-cached) portion of the system prompt so the model responds in the user's chosen language for the entire session.

**Where:** `services/llm.py` → `_LANG_INSTRUCTIONS` + `chat_response()`

```python
_LANG_INSTRUCTIONS = {
    "en": "",
    "es": "\nLANGUAGE: Respond in Spanish (Español). Keep the same warm, concise style.",
    "fr": "\nLANGUAGE: Respond in French (Français). Keep the same warm, concise style.",
    "hi": "\nLANGUAGE: Respond in Hindi (हिंदी). Keep the same warm, concise style.",
    ...
}

# In chat_response():
lang_instr = _LANG_INSTRUCTIONS.get(language, "")
dynamic_system = f"\nCurrent booking context:\n{ctx_str}{lang_instr}"
```

**Why in dynamic, not static:** The static block is cached as one shared copy. If the language instruction was in the static block, you'd need 8 separate cached copies (one per language). By putting it in the dynamic block, all 8 languages share the same cached static rules.

**Detection:** `language_selection_node` detects the language via a keyword map (e.g. "2" or "español" → "es") with a Haiku fallback for ambiguous inputs.

---

## 12. Feature Deep Dives

### Multi-Language Flow (End to End)

```
1. New session → conv_state = "SELECTING_LANGUAGE"
2. Frontend sends empty text → main.py returns language menu
3. User says "2" or "español" or "spanish"
4. language_selection_node:
   a. _detect_language("2") → "es"  (keyword map hit)
   b. state["language"] = "es"
   c. Response in Spanish: "¡Perfecto! Te ayudaré en español..."
   d. conv_state → "COLLECTING_PHONE"
5. All subsequent chat_response() calls include:
   "\nLANGUAGE: Respond in Spanish (Español)."
6. Frontend updates ttsLang → browser speaks Spanish TTS
```

### CoT Flight Analysis Flow

```
1. search_flights_node fetches flights from external API
2. _rank_flights() does a quick heuristic sort (price + duration * 0.5)
3. cot_analyze_flights() sends structured prompt to Haiku:
   - Lists all flights with price, duration, stops
   - Asks for THINKING/BEST_VALUE/CHEAPEST/FASTEST/SUMMARY
4. _apply_cot_tags() marks each flight dict with booleans:
   { ..., "best_value": True, "cheapest": False, "fastest": False }
5. Flights returned to frontend with these tags
6. Frontend renders colored badges on each flight card
7. CoT summary injected into the task for the spoken response
```

### Login / Auth Dual Path

```
Voice call path (no account):
  → OTP sent via Twilio to phone
  → User reads back 6-digit code
  → auth_node verifies code
  → authenticated = True

Web UI path (has account):
  → User logs in via modal before starting chat
  → user_id injected into every /api/voice request
  → main.py: if req.user_id and not state.get("user_id"): state["user_id"] = req.user_id
  → OTP step still runs (simplification — could be skipped for web-authenticated users)
```

### LLM Judge Background Thread

```
main.py receives user turn
  → graph.invoke() runs (blocking, up to 15s)
  → response sent to user  ← user gets answer here
  → threading.Thread(target=_run_judge_bg, daemon=True).start()
      → judge_response() calls Haiku (blocking, ~1s)
      → parse scores
      → save_judge_log() writes to SQLite
      → thread exits
```

---

## 13. Request Lifecycle — End to End

Here's exactly what happens from "user sends a message" to "user gets a response":

```
1. Browser: POST /api/voice
   { text: "I want to fly to New York", session_id: "abc", language: "en" }

2. main.py:
   a. load_session("abc") → loads state from SQLite
   b. state["user_input"] = "I want to fly to New York"

3. timeout_guard: calls graph.invoke(state) with 15s timeout

4. graph.invoke(state):
   a. security_guard_node:
      - Length check: 34 chars ✓
      - Regex: no injection ✓
      - Haiku classifier: "NO" (safe) ✓
      - Returns: {blocked: False}

   b. after_security → "intent_router"

   c. intent_router_node:
      - Haiku: "What is the intent?" → "normal"
      - conv_state is "COLLECTING_DESTINATION"
      - Returns: {intent: "normal"}

   d. should_continue → "resolve_airport"

   e. resolve_airport_node:
      - Extracts "New York" from input
      - database.resolve_airport("New York") → {iata: "JFK", city: "New York"}
      - chat_response("User said New York (JFK). Confirm destination...") → "Got it! New York JFK. ..."
      - Returns: {destination_city: "New York", destination_iata: "JFK", conv_state: "COLLECTING_DATE", response: "..."}

   f. graph → END

5. main.py:
   a. new_conv_state = "COLLECTING_DATE"
   b. save_session("abc", "COLLECTING_DATE", result)
   c. threading.Thread(target=_run_judge_bg, ...).start()  ← background

6. Return VoiceResponse:
   { response: "Got it! New York JFK. What date would you like to travel?",
     conv_state: "COLLECTING_DATE", suggestions: [], flights: [] }

7. Browser:
   a. Displays response text
   b. speak() → browser TTS reads it aloud
   c. Updates step indicator to "Date"
```

Total latency: ~1-2 seconds (security guard Haiku + intent router Haiku + resolve_airport response Haiku + hallucination guard Haiku, all sequential with 8s per-call timeout).
