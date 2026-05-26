# Phoenix Air — Agent Flow

---

## 1. Top-Level Pipeline

```mermaid
flowchart TD
    START([User Input]) --> SG

    SG["🛡️ security_guard\nlength + regex + LLM classifier\n(3-layer)"]
    SG -->|blocked| ENDB([END — end_call = true])
    SG -->|safe| IC

    IC["🔍 input_classifier\nGroq → Haiku fallback\nclassifies: answer / question / query"]
    IC -->|question or query| IR
    IC -->|answer| AV

    AV["✅ answer_validator"]
    AV -->|Step 0 — global intent override\nLLM detects mis-labelled greeting /\ntransfer / policy inside an answer| IR
    AV -->|pending correction: YES| IR
    AV -->|pending correction: NO| CLRP([ask again → END])
    AV -->|L1 regex ✓ valid| IR
    AV -->|L1 regex ✗, L2 LLM corrects it| CORR([Did you mean X? → END])
    AV -->|L1 + L2 both fail| HINT([I didn't catch that → END])

    IR["🧭 intent_router\nkeyword → embedding → LLM\n(3-layer hybrid)"]
    IR -->|greeting| GN
    IR -->|policy| RP
    IR -->|general_qa| RP
    IR -->|out_of_domain| OOD
    IR -->|field_correction| FC
    IR -->|transfer| TR
    IR -->|customer_contact| CC
    IR -->|restart| RA
    IR -->|normal — route by conv_state| CS

    CS{conv_state?}
    CS -->|IDLE\nCOLLECTING_DEPARTURE\nCOLLECTING_DESTINATION| RA
    CS -->|COLLECTING_DATE| SF
    CS -->|PRESENTING_FLIGHTS| PF
    CS -->|COLLECTING_PASSENGER| CP
    CS -->|COLLECTING_CONTACT| BF
    CS -->|COLLECTING_PAYMENT| PAY
    CS -->|COLLECTING_PHONE\nVERIFYING_OTP| AUTH
    CS -->|SELECTING_LANGUAGE| LS
    CS -->|POST_BOOKING| GN
    CS -->|BOOKING_CONFIRMED\nDONE| ENDC([END — end_call = true])

    GN["👋 greeting_node\nLLM — adapts to booking stage"]
    OOD["🚫 out_of_domain_node\nHARDCODED — fixed string\n'I specialise in airline booking...'"]
    FC["✏️ field_correction_node\nkeyword → LLM field detect\nclears downstream state"]
    TR["☎️ transfer_node\nhardcoded transfer message"]
    CC["📞 customer_contact_node\nhardcoded support details"]
    RA["🛫 resolve_airport\nIATA lookup + fuzzy suggest"]
    SF["🔎 search_flights\nmulti-date parallel search\nheap ranking"]
    PF["📋 present_flights\nnumber / ordinal / airline-name pick"]
    CP["🧑 collect_passenger\nfirst + last name parse"]
    BF["📝 book_flight\nAPI POST + SQLite + send_confirmation"]
    PAY["💳 payment\nmock_charge + PDF ticket + email"]
    AUTH["🔐 auth\nOTP generate → Twilio SMS → verify"]
    LS["🌐 language_selection\nsets language in state"]
    RP["📚 rag_policy\nsee Section 2"]

    GN --> ENDA([END])
    OOD --> ENDA
    FC --> ENDA
    TR --> ENDA
    CC --> ENDA
    RA --> ENDA
    SF --> ENDA
    PF --> ENDA
    CP --> ENDA
    BF --> ENDA
    PAY --> ENDA
    AUTH --> ENDA
    LS --> ENDA
```

---

## 2. rag_policy — Internal Layers

Handles both `policy` intent and `general_qa` intent.

```mermaid
flowchart TD
    ENTRY([rag_policy entered])

    ENTRY --> CHK{intent == general_qa?}

    CHK -->|No — run policy path first| PL1
    CHK -->|Yes — skip policy, go straight to airline check| AIRCHK

    PL1["Policy L1\nkeyword match → hardcoded short answer"]
    PL1 -->|match found| PL1R([hardcoded answer + 'Anything else?' → END])
    PL1 -->|no match| PL2

    PL2["Policy L2\nTF-IDF cosine retrieval → LLM\ngrounded to policy document\n+ hallucination guard"]
    PL2 -->|guard passes| PL2R([LLM answer grounded to policy → END])
    PL2 -->|guard fails or no doc| AIRCHK

    AIRCHK{Is input airline-related?\nkeyword list check}
    AIRCHK -->|NOT airline-related| FOOD
    AIRCHK -->|airline-related| GL1

    FOOD(["🚫 _FIXED_OOD hardcoded\n'I specialise in flight booking...\nI'm not able to help with that' → END"])

    GL1["General L1\nkeyword match → hardcoded general answer"]
    GL1 -->|match found| GL1R([hardcoded answer + 'Anything else?' → END])
    GL1 -->|no match| GL2

    GL2["General L2\nTF-IDF cosine on general knowledge docs\n→ LLM grounded to doc"]
    GL2 -->|doc found| GL2R([LLM answer grounded to general doc → END])
    GL2 -->|no doc| GL3

    GL3["General L3\nDuckDuckGo web search\n→ LLM grounded to results"]
    GL3 -->|results found| GL3R([LLM answer grounded to web → END])
    GL3 -->|no results| GL4

    GL4["General L4\nLLM from training knowledge only"]
    GL4 -->|response generated| GL4R([LLM answer → END])
    GL4 -->|LLM fails| GL5

    GL5["General L5\nEscalate — set intent = customer_contact"]
    GL5 --> CCN([customer_contact_node → support details → END])
```

---

## 3. answer_validator — Internal Steps

Runs only when `input_classifier` returns `answer`.

```mermaid
flowchart TD
    AV_ENTRY(["answer_validator entered\ninput already labelled 'answer'"])

    AV_ENTRY --> PC{pending_correction\nstored from previous turn?}

    PC -->|Yes — user said YES / confirm| PC_YES["apply corrected value as user_input\nvalidated = True"]
    PC -->|Yes — user said NO / reject| PC_NO(["'Could you provide X again?'\nvalidated = False → END"])
    PC -->|Yes — user gave new input| PC_NEW["clear pending_correction\ncontinue to Step 0"]
    PC -->|No pending| STEP0

    PC_YES --> IR_OUT([→ intent_router])
    PC_NEW --> STEP0

    STEP0["Step 0 — Global Intent Override\nCatches mis-labelled greetings,\ntransfers, policy questions"]

    STEP0 --> SC{Zero-ambiguity\nkeyword match?}
    SC -->|Yes\n'hello', 'start over',\n'speak to a human', '?'…| REROUTE["update input_type\nto query or question\nvalidated = True"]
    SC -->|No| LLM0

    LLM0["LLM check\nGroq → Haiku\n'Is this actually a question or query?'"]
    LLM0 -->|Yes — it's a question or query| REROUTE
    LLM0 -->|No — it really is an answer| STATECHECK

    REROUTE --> IR_OUT2([→ intent_router])

    STATECHECK{conv_state needs\nvalidation?}
    STATECHECK -->|No — state not in validation list\ne.g. COLLECTING_PHONE, auth states| PASSTHRU["validated = True"]
    STATECHECK -->|Yes| STEP1
    PASSTHRU --> IR_OUT3([→ intent_router])

    STEP1["Step 1 — L1 Regex Validation\nper conv_state"]

    STEP1 --> R1{Which state?}
    R1 -->|COLLECTING_DEPARTURE\nCOLLECTING_DESTINATION| R_CITY["city regex\nletters, spaces, hyphens only\nmin 2 chars"]
    R1 -->|COLLECTING_DATE| R_DATE["date pattern list\n2026-08-15 / Aug 15 2026\nnext Monday / tomorrow…"]
    R1 -->|COLLECTING_PASSENGER| R_NAME["name regex\nletters + spaces\nmust have ≥ 2 words"]
    R1 -->|COLLECTING_CONTACT| R_CONTACT["email regex OR\nphone regex\n7–20 digits"]
    R1 -->|PRESENTING_FLIGHTS| R_FLIGHT["single digit 1–5"]

    R_CITY & R_DATE & R_NAME & R_CONTACT & R_FLIGHT --> REGEXRESULT{regex\npasses?}

    REGEXRESULT -->|Yes — date input| DATE_NORM["LLM normalises to YYYY-MM-DD\ne.g. 'next Tuesday' → '2026-05-26'"]
    REGEXRESULT -->|Yes — non-date| V_TRUE["validated = True"]
    DATE_NORM --> IR_OUT4([→ intent_router])
    V_TRUE --> IR_OUT5([→ intent_router])

    REGEXRESULT -->|No| STEP2

    STEP2["Step 2 — L2 LLM Interpretation\nGroq → Haiku\nspelling correction + natural-language parse"]

    STEP2 --> L2R{LLM result?}

    L2R -->|Corrected + confident\ne.g. 'Los Angles' → 'Los Angeles'| CONF["store as pending_correction\n'Did you mean Los Angeles?'\nvalidated = False → END"]
    L2R -->|Corrected + uncertain\ne.g. 'Jhn Smth' → 'John Smith'| UNCONF["store as pending_correction\n'I'm not sure — did you mean John Smith?'\nvalidated = False → END"]
    L2R -->|Cannot interpret| BADINPUT(["regex hint message\n'Please provide a valid city name.'\nvalidated = False → END"])
```

---

## 4. Conv State → Node Routing (intent = normal)

| Conv State | Node |
|---|---|
| `IDLE` | resolve_airport |
| `SELECTING_LANGUAGE` | language_selection |
| `COLLECTING_PHONE` | auth |
| `VERIFYING_OTP` | auth |
| `COLLECTING_DEPARTURE` | resolve_airport |
| `COLLECTING_DESTINATION` | resolve_airport |
| `COLLECTING_DATE` | search_flights |
| `PRESENTING_FLIGHTS` | present_flights |
| `COLLECTING_PASSENGER` | collect_passenger |
| `COLLECTING_CONTACT` | book_flight |
| `COLLECTING_PAYMENT` | payment |
| `POST_BOOKING` | greeting |
| `BOOKING_CONFIRMED` | end_node |
| `DONE` | end_node |

---

## 5. Response Type Labels

| Label | Meaning |
|---|---|
| `hardcoded` | Fixed string, no LLM involved |
| `llm` | LLM-generated response |
| `fallback` | Timeout or error fallback message |

---

## 6. LLM Provider Priority

Every LLM call in the system tries **Groq (Llama 3.1 8B)** first (~100ms), then falls back to **Claude Haiku** (~600ms) if Groq is unavailable or errors. The `GROQ_API_KEY` env var controls this — if blank, all calls go directly to Haiku.
