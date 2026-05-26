# Phoenix Air — What You Built & What to Learn Next

## What You Built

A production-style voice/text airline booking agent with:
- FastAPI backend + LangGraph state machine
- 3-layer hybrid intent router (keyword → sentence embeddings → LLM)
- Phone OTP authentication
- External flight API integration with parallel multi-date search
- Mock payment system + PDF ticket generation
- Interactive flight card UI with client-side sort
- Hallucination guard + timeout guard
- SQLite for sessions, bookings, users, OTP, payments

---

## Concepts You Have Learned

### 1. LLM Integration
- How to call Claude Haiku via the Anthropic SDK
- Grounded generation: pass a retrieved document as context so the LLM answers from facts, not hallucination
- Prompt design: system prompt sets role + constraints, user message carries the question

### 2. Retrieval-Augmented Generation (RAG)
- TF-IDF keyword matching to find the most relevant policy chunk
- Passing that chunk to an LLM instead of letting it answer freely
- Why grounding matters: without it the LLM invents policy details

### 3. Hybrid Intent Routing
- Layer 1 (keyword scan): fast, zero-cost, handles the obvious cases
- Layer 2 (sentence embeddings): `all-MiniLM-L6-v2` encodes text into vectors; cosine similarity measures semantic closeness — catches paraphrases keyword lists miss
- Layer 3 (LLM fallback): when both layers are uncertain, ask the LLM to classify
- Why layering matters: each layer is slower and more expensive than the last — you only pay the expensive cost when cheaper methods fail

### 4. State Machines for Conversations
- LangGraph models a conversation as a graph of nodes connected by conditional edges
- Each node reads `state`, adds its output, returns updated `state`
- `should_continue()` (the router) picks the next node based on `intent` + `conv_state`
- This is more predictable than a pure LLM agent that decides its own next action

### 5. Embedding-Based Similarity
- Sentence transformers convert text into float vectors
- Cosine similarity = dot product of two unit vectors — measures how "close" two meanings are
- Pre-computing anchor vectors at startup saves time on every request
- Threshold (0.45) controls how confident you need to be before acting on the result

### 6. API Design
- Single POST endpoint (`/api/voice`) handles the entire conversation
- Stateless endpoint + stateful SQLite session = each request is self-contained but the conversation is remembered
- Response schema carries everything the frontend needs: `response`, `conv_state`, `flights`, `end_call`

### 7. Date Parsing
- Three-layer approach: fast regex → smart library (dateutil) → LLM fallback
- `dateutil fuzzy=True` is powerful but too permissive — always guard it with a content check
- Year inference: if no year given and the date is already past, assume next year

### 8. Authentication
- OTP flow: generate random 6-digit code → store in DB with expiry → validate on input → mark used
- Why mark OTP as used after first verify: prevents replay attacks
- Dev mode pattern: if third-party (Twilio) not configured, show the code in-app so development doesn't break

### 9. Guards
- **Hallucination guard**: a second LLM call that checks if the first answer is supported by the source document. Cost: one extra Haiku call per policy answer.
- **Timeout guard**: `ThreadPoolExecutor.submit().result(timeout=N)` wraps any slow function — on timeout, return a safe fallback instead of crashing

### 10. PDF Generation + Email
- `reportlab` builds PDFs programmatically (canvas, strings, lines)
- `smtplib` + `MIMEMultipart` attaches the PDF bytes to an email
- Pattern: generate bytes in memory → attach → send. No temp files needed.
---
## What to Learn Next (Remaining Features)

### 1. Observability
**What it is:** Knowing what your system is doing in production — which nodes ran, how long each took, what the LLM returned, where errors happened.

**How to approach:**
- Add a `request_id` (UUID) to every `/api/voice` call and pass it through the graph
- Log structured JSON lines: `{"request_id": ..., "node": ..., "duration_ms": ..., "intent": ...}`
- Use Python `logging` with a JSON formatter, or a tool like **Loguru**
- For LLM calls specifically, log: model, prompt token count, completion token count, latency
- Later: send logs to a service like **Datadog**, **Grafana Loki**, or just a file

**Key concept:** structured logs (JSON) are searchable; plain print() is not.

---

### 2. Prompt Caching
**What it is:** Anthropic's API lets you mark part of a prompt as "cached" — if the same prefix is reused, they don't re-process it, saving cost and latency.

**How to approach:**
- The system prompt and the policy document are the same on every call — mark them with `"cache_control": {"type": "ephemeral"}`
- Only the user's message changes each turn — that goes uncached
- Saves ~90% of input token cost for repeated system prompts
- Read: [Anthropic prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

```python
# Example structure
messages=[
    {"role": "user", "content": [
        {"type": "text", "text": policy_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": user_question}
    ]}
]
```

---

### 3. LLM as Judge
**What it is:** After generating a response, a second LLM evaluates it on criteria like helpfulness, accuracy, and tone. Different from hallucination guard — it judges quality, not just factual grounding.

**How to approach:**
- After every node response, optionally call Haiku with: "Rate this response 1-5 for helpfulness and accuracy. Reply JSON: `{score, reason}`"
- If score < 3, regenerate or flag for review
- This is the foundation of **RLHF-style self-improvement** — you collect judge scores and use them to improve prompts over time
- Start simple: log scores to a `response_quality` table in SQLite

---

### 4. Chain of Thought + Self-Reflection
**What it is:** Instead of asking the LLM to answer directly, ask it to think step-by-step first (CoT), then check its own answer (self-reflection).

**How to approach for flight selection:**
```
System: You are a flight recommendation assistant. Think step by step.
User: The user wants the best value flight. Here are the options: [flights].
      Step 1: Rank by value (price + duration).
      Step 2: Check if your top pick makes sense.
      Step 3: Give a one-sentence recommendation.
```
- Self-reflection: after generating a recommendation, add "Now review your answer. Is it correct? If not, correct it."
- CoT adds ~200 tokens of reasoning but significantly improves accuracy on multi-step decisions

---

### 5. Security Guard (Input Sanitisation)
**What it is:** Checking user input before it enters the system — blocking prompt injection, XSS, SQL injection, and abuse attempts.

**How to approach:**
- Layer 1: length limit (reject inputs > 500 chars)
- Layer 2: regex blocklist — detect SQL keywords, script tags, prompt injection patterns (`ignore previous instructions`, `you are now`, etc.)
- Layer 3: Haiku classifier — "Is this message a prompt injection attempt or abuse? Reply YES or NO."
- Add this as the very first node in the graph, before `intent_router`

---

### 6. Discount System for Returning Users
**What it is:** `user_profiles` already stores `call_count`. When a returning user books, offer a discount.

**How to approach:**
- In `auth_node` after OTP verification, query `user_profiles` by phone
- If `call_count >= 2`, set `state["discount"] = 0.10` (10%)
- In `payment_node`, apply: `final_price = flight_price * (1 - discount)`
- Tell the user: "Welcome back! As a returning customer you get 10% off."
- Update `call_count += 1` on every successful booking

---

### 7. Multi-Language Support
**What it is:** Detecting the user's language and responding in the same language.

**How to approach:**
- Add a language detection step in `intent_router` using `langdetect` (pip package, fast, rule-based)
- Store `state["language"]` = `"es"`, `"fr"`, `"en"`, etc.
- In every node, append to the system prompt: `"Respond in {language}."`
- For flight cards in the UI: use `Intl.NumberFormat` for currency in the user's locale
- Hardest part: month names and date formats vary by locale — the LLM handles this naturally if you tell it the language

---

### 8. Full Call Transfer (Phonely Integration)
**What it is:** When `transfer=True`, actually route the caller to a live agent via phone.

**How to approach:**
- Phonely webhook: when your API returns `"transfer": true`, Phonely will call a second number
- In your `.env`, set `TRANSFER_NUMBER=+1800XXXXXXX`
- Your `/api/voice` response already has `transfer: bool` — Phonely reads this and acts on it
- No code change needed in the backend; this is a Phonely dashboard configuration

---

## The Learning Path (Recommended Order)

```
NOW (basics solid)
  │
  ├── Observability first — you can't improve what you can't measure
  ├── Prompt caching — immediate cost saving, 1 hour to add
  ├── Security guard — before any real users
  ├── Discount system — builds on existing user_profiles table
  │
INTERMEDIATE
  ├── LLM as judge — start logging scores, don't act on them yet
  ├── CoT + self-reflection — experiment on flight recommendation node
  │
ADVANCED
  ├── Multi-language — language detection is easy; testing is hard
  └── Full call transfer — requires Phonely production setup
```

---
  
## Key Mental Models to Keep

1. **Cheap first** — always run keyword scan before embeddings, embeddings before LLM. Each step is 10-100x more expensive.
2. **State machine > agent** — for structured flows (booking), a state machine is more predictable and debuggable than a free-form LLM agent.
3. **Guard at the boundary** — validate input at entry (security guard), validate output before return (hallucination guard). Never trust the middle.
4. **Dev mode everything** — for any paid/external service (Twilio, SMTP, Stripe), always have a dev fallback that prints to console so development never breaks.
5. **Log before you optimize** — add observability before you try to improve performance. Otherwise you're guessing.
