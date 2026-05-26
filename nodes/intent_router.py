"""
Intent Router — Layer 2 in the pipeline (runs after input_classifier).

Uses input_type set by input_classifier to decide routing:

  answer   -> intent = "normal"  (route by conv_state, skip all detection)
  question -> policy / general_qa / out_of_domain detection
               L1: keyword match
               L2: OOD guard
               L3: embedding similarity
               L4: LLM classification (always runs if still undecided)
  query    -> full intent detection pipeline
               L1: keyword match (transfer / restart / greeting / field_correction /
                                   customer_contact / general_qa / policy)
               L2: OOD guard
               L3: embedding similarity
               L4: LLM fallback

Note: HumanMessage is added to state.messages by input_classifier, NOT here.
"""

import os
import numpy as np


TRANSFER_KEYWORDS = [
    "transfer", "agent", "human", "representative", "customer support",
    "customer service", "speak to someone", "talk to someone",
    "real person", "live agent",
]

POLICY_KEYWORDS = [
    # refund / money
    "refund", "money back", "reimburse", "reimbursement", "get my money",
    # cancellation
    "cancel", "cancellation", "no show", "no-show", "forfeit",
    # flight change
    "change flight", "change my flight", "reschedule", "modify my flight",
    "different flight", "move my flight", "switch flight",
    # check-in
    "check in", "check-in", "checkin", "check in time", "check in deadline",
    "when should i arrive", "how early", "boarding time", "boarding pass",
    "gate closes",
    # baggage
    "baggage", "luggage", "bag", "suitcase", "carry on", "carry-on",
    "checked bag", "extra bag", "oversize", "overweight", "sports equipment",
    "bag fee", "baggage fee", "baggage allowance", "baggage limit",
    # seat
    "seat", "seat selection", "change seat", "pick seat", "choose seat",
    "seat upgrade", "exit row", "bulkhead", "window seat", "aisle seat",
    "extra legroom",
    # upgrade / class
    "upgrade", "business class", "first class", "premium cabin",
    # general policy words
    "policy", "policies", "rules", "terms", "conditions", "fee", "fees",
    # credit / voucher
    "travel credit", "voucher", "credit", "non-refundable",
    # missed flight
    "miss my flight", "missed flight", "missed my flight", "no show",
    # delay / disruption (policy questions, not complaints)
    "delay", "delayed", "was delayed", "flight delay", "delay compensation", "delay policy",
    "what happens if", "what if my flight",
    # pets / animals
    "pet", "dog", "cat", "service animal", "emotional support animal",
    "fly with pet", "travel with pet", "bring my pet", "bring my dog", "bring my cat",
    # loyalty / miles
    "miles", "loyalty", "frequent flyer", "phoenix miles", "earn miles", "redeem miles",
    "loyalty program", "rewards program", "points program",
    # payment methods
    "paypal", "apple pay", "google pay", "payment method", "payment plan",
    "accepted cards", "pay with", "installment",
    # name correction
    "name change", "name correction", "wrong name", "typo in name",
    "correct my name", "fix my name", "spelling mistake on ticket",
    "misspelled name", "change name on ticket",
    # overbooking / denied boarding
    "overbooked", "overbooking", "bumped from", "denied boarding",
    "volunteer to give up", "give up my seat", "involuntary denied",
    # medical / pregnancy
    "pregnant", "pregnancy", "flying pregnant", "weeks pregnant",
    "pacemaker", "cpap", "cpap machine", "fly after surgery",
    "portable oxygen concentrator", "cast on flight", "doctor note to fly",
    # prohibited items
    "prohibited item", "forbidden item", "not allowed on plane",
    "firearms on flight", "gun in luggage", "lithium battery",
    "hoverboard on plane", "e-cigarette on plane", "vape on plane",
    "dangerous goods", "what can i not bring",
    # group bookings
    "group booking", "group travel", "group discount", "group fare",
    "book for a group", "corporate booking", "corporate travel",
]

RESTART_KEYWORDS = [
    "start over", "restart", "begin again", "start again", "reset",
    "different flight", "book another",
]

GREETING_KEYWORDS = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "what's up", "sup",
]

FIELD_CORRECTION_KEYWORDS = [
    "change my departure", "change departure", "change my destination",
    "change destination", "change the date", "change my date",
    "update departure", "update destination", "update date",
    "wrong city", "wrong date", "actually i meant", "i meant",
    "go back to", "change passenger", "change my name", "change contact",
    "change my contact", "change the city", "different departure",
    "different destination", "different date",
    # "update my <field>" variants — previously missing, fell through to embedding/LLM
    "update my phone", "update phone", "update my email", "update email",
    "update my name", "update my contact", "update my passenger",
    "update my departure", "update my destination", "update my date",
    "update name", "update contact",
]

GENERAL_QA_KEYWORDS = [
    # destinations / routes
    "do you fly", "do you go to", "which destinations", "what destinations",
    "what cities do you", "which cities", "what routes", "do you have flights to",
    "international routes", "domestic routes", "fly internationally", "fly to",
    # aircraft / fleet
    "what aircraft", "what plane", "what fleet", "boeing", "airbus", "dreamliner",
    "type of plane", "kind of aircraft",
    # pets
    "bring my pet", "bring my dog", "bring my cat", "travel with pet",
    "fly with pet", "can i take my pet", "pet on the plane", "pet in cabin",
    # wifi / internet
    "wifi", "wi-fi", "internet on the flight", "in-flight wifi", "is there wifi",
    "wifi available", "online during flight",
    # meals / food
    "what meals", "food on the flight", "meal service", "in-flight meal",
    "snacks on", "is there food", "do you serve food", "what do you serve",
    "drinks on", "beverages on",
    # loyalty / miles
    "loyalty program", "frequent flyer", "phoenix miles", "earn miles",
    "redeem miles", "rewards program", "how do i earn", "points program",
    # special assistance
    "wheelchair", "special assistance", "disability", "medical needs",
    "mobility aid", "need assistance", "accessible",
    # children / infants
    "unaccompanied minor", "child flying alone", "infant on", "baby on the flight",
    "travelling with infant", "child fare", "kid on the plane",
    # flight duration / arrival time
    "how long is the flight", "flight duration", "how early to arrive",
    "when to arrive", "how long does it take", "flight time",
    # entertainment
    "in-flight entertainment", "entertainment on", "movies on", "tv on the flight",
    "music on the flight", "screens on the plane",
    # cabin / class options
    "cabin options", "class options", "economy class", "what classes",
    # payment methods
    "apple pay", "paypal", "payment methods", "do you accept", "credit card",
    "debit card", "how do i pay", "payment options",
    # travel documents
    "need passport", "need id", "what documents", "visa required",
    "passport required", "travel documents",
    # flight status / tracking
    "flight status", "track my flight", "is my flight on time", "flight delay",
    "on time", "flight tracker",
    # booking tips / pricing
    "cheapest time to book", "cheapest day to fly", "best time to book",
    "best time to fly", "when to book", "how far in advance", "cheapest flight",
    "save money on flights", "best deal on flights", "cheapest fare",
    "when are flights cheapest", "when is the cheapest",
    # airport / security / boarding
    "airport security", "security check", "security rules", "what can i bring",
    "liquids on the plane", "tsa", "boarding process", "boarding order",
    "how does boarding work", "when does boarding start",
    # connections / layovers
    "connecting flight", "miss my connection", "layover", "transit",
    "stopover", "connection time",
    # international travel
    "international travel", "what documents", "do i need a visa",
    "do i need a passport", "travel documents",
    # entertainment
    "movies on the plane", "movies on the flight", "in-flight entertainment",
    "seatback screen", "what to watch",
    # airport arrival
    "how early should i", "how early to arrive", "when should i arrive at",
    "how much time before", "arrive at the airport",
    # travel insurance
    "travel insurance", "trip insurance", "flight insurance",
    # health / wellbeing on flights
    "jet lag", "time zone adjustment", "dvt", "blood clot", "deep vein",
    "ear pressure", "ear pain on plane", "popping ears",
    "dehydration on flight", "motion sickness on plane",
    "fear of flying", "flying anxiety", "scared to fly",
    # aircraft safety
    "defibrillator", "aed", "cardiac arrest on plane", "heart attack on plane",
    "oxygen mask", "emergency exit", "brace position",
    "cabin pressure", "cabin depressurisation",
    "dim the lights", "cabin lights during takeoff", "why lights dimmed",
    "turbulence", "plane shaking", "rough flight",
    "is flying safe", "aviation safety",
    # airport facilities
    "airport lounge", "duty free", "duty-free", "airport wifi",
    "airport shop", "airport restaurant", "currency exchange airport",
    "left luggage", "airport pharmacy",
    # first-time flyer
    "first time flying", "first time flyer", "never flown before",
    "what to expect on a flight", "new to flying",
]

CUSTOMER_CONTACT_KEYWORDS = [
    "complaint", "complain", "feedback", "issue", "problem",
    "lost baggage", "lost bag", "damaged bag", "damaged luggage",
    "delay", "delayed", "flight delayed", "flight cancelled",
    "billing issue", "overcharged", "charged wrongly", "wrong charge",
    "report", "file a complaint", "raise a complaint",
    "not happy", "unhappy", "dissatisfied", "frustrated",
    "terrible service", "bad experience", "worst experience",
    "missing bag", "bag not arrived", "bag lost",
]

_OOD_TRIGGERS = [
    "what is", "what are", "what was", "who is", "who are", "tell me",
    "explain", "can you tell", "do you know", "how do i make",
    "where is", "where are", "why is", "why are", "what's the",
    "describe", "define",
]
_AIRLINE_TERMS = [
    "flight", "book", "cancel", "airport", "airline", "travel",
    "seat", "bag", "refund", "ticket", "fly", "depart", "arriv",
    "trip", "route", "destination", "check in", "checkin",
    "board", "gate", "luggage", "change", "reschedule",
]


INTENT_ANCHORS = {
    "transfer": [
        "speak to a human",
        "talk to an agent",
        "customer service representative",
        "I need to speak with someone",
        "connect me to a live person",
        "can I talk to a real person",
        "transfer me to support",
    ],
    "policy": [
        "what is your refund policy",
        "how much does an extra bag cost",
        "what is the cancellation fee",
        "what is the check-in deadline",
        "can I change my seat",
        "what happens if I miss my flight",
        "is there a fee for changing my flight",
        "what is the baggage allowance",
        "how do I cancel my booking",
        "what are the rules for carry-on bags",
        "will I get a refund if I cancel",
        "how much does seat upgrade cost",
        "when does online check-in open",
        "what is the no-show policy",
        "can I get a travel credit",
    ],
    "restart": [
        "start over please",
        "book a different flight",
        "begin again",
        "I want to start a new booking",
        "can we restart",
        "let me book another flight",
        "I want to change everything",
    ],
    "greeting": [
        "hello there",
        "hi good morning",
        "hey how are you",
        "good evening",
        "howdy",
        "hi there",
    ],
    "field_correction": [
        "I want to change my departure city",
        "I meant a different date",
        "update my destination",
        "actually I want to go somewhere else",
        "I made a mistake with the date",
        "can we change the city",
        "I typed the wrong city",
        "let me correct my departure",
        "I want a different destination",
    ],
    "customer_contact": [
        "I have a complaint",
        "my bag was damaged",
        "I was overcharged on my ticket",
        "I have an issue with my booking",
        "something went wrong with my flight",
        "I need help with a problem",
        "I want to report an issue",
        "my flight was delayed and I need help",
        "I am not happy with the service",
    ],
    "out_of_domain": [
        "what is the capital of France",
        "tell me a joke",
        "who won the game last night",
        "what is the weather like today",
        "how do I cook pasta",
        "who is the president",
        "what is two plus two",
        "tell me about world history",
        "recommend a movie",
    ],
    "general_qa": [
        "do you fly to London",
        "can I bring my dog on the plane",
        "is there wifi on the flight",
        "what meals are served",
        "do you have a loyalty program",
        "how long is the flight",
        "which cities do you serve",
        "what aircraft do you use",
        "do you offer wheelchair assistance",
        "can infants fly on your airline",
        "are there entertainment options on board",
        "what payment methods do you accept",
        "do you fly internationally",
        "how early should I arrive at the airport",
        "do I need a passport for domestic flights",
        "is there food served on the plane",
        "can I track my flight",
    ],
    "normal": [
        "I want to fly to New York",
        "book a flight for August",
        "Los Angeles please",
        "I need a ticket to Paris",
        "one way to Chicago",
        "flying to Miami next month",
        "I want to travel to London",
        "find me a flight",
    ],
}

_embedder = None
_anchor_vectors: dict = {}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_anchor_vectors():
    global _anchor_vectors
    if not _anchor_vectors:
        embedder = _get_embedder()
        for intent, sentences in INTENT_ANCHORS.items():
            vecs = embedder.encode(sentences, normalize_embeddings=True)
            _anchor_vectors[intent] = np.mean(vecs, axis=0)
    return _anchor_vectors


def _embedding_intent(text: str) -> tuple[str, float]:
    """Return (best_intent, confidence) using cosine similarity."""
    try:
        embedder = _get_embedder()
        anchors = _get_anchor_vectors()
        q_vec = embedder.encode([text], normalize_embeddings=True)[0]
        best_intent = "normal"
        best_score = -1.0
        for intent, anchor_vec in anchors.items():
            score = float(np.dot(q_vec, anchor_vec))
            if score > best_score:
                best_score = score
                best_intent = intent
        return best_intent, best_score
    except Exception:
        return "normal", 0.0


def _llm_intent(text: str) -> str:
    """Ask Claude Haiku to classify intent across all intent types — full pipeline fallback."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "normal"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        valid = ", ".join(INTENT_ANCHORS.keys())
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=16,
            system=(
                f"Classify the airline chatbot user message into exactly one of: {valid}. "
                "Reply with ONLY the intent name, nothing else."
            ),
            messages=[{"role": "user", "content": text}],
        )
        result = msg.content[0].text.strip().lower()
        return result if result in INTENT_ANCHORS else "normal"
    except Exception:
        return "normal"


def _llm_check_greeting(text: str) -> bool:
    """
    Detect whether input is a greeting or casual social opener.
    Tries Groq first (fast), falls back to Haiku.
    Used as the last resort before declaring out_of_domain.
    """
    system = (
        "Is this message a greeting, farewell, or casual social opener "
        "(e.g. 'hello', 'hi there', 'good morning', 'thanks', 'bye', 'how are you')? "
        "Reply with ONLY 'yes' or 'no'."
    )

    # Primary: Groq
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from groq import Groq
            raw = Groq(api_key=groq_key).chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": text},
                ],
                max_tokens=3,
                temperature=0,
            ).choices[0].message.content.strip().lower()
            return raw.startswith("yes")
        except Exception as e:
            print(f"[INTENT] Groq greeting check error: {e}")

    # Fallback: Haiku
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return False
    try:
        import anthropic
        raw = anthropic.Anthropic(api_key=api_key).messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3,
            system=system,
            messages=[{"role": "user", "content": text}],
        ).content[0].text.strip().lower()
        return raw.startswith("yes")
    except Exception:
        return False


def _llm_classify_question(text: str) -> str:
    """
    Classify a 'question' type input as policy / general_qa / out_of_domain.
    Called when keyword and embedding layers haven't resolved it.
    LLM is always the final decision-maker for question-type inputs.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "general_qa"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            system=(
                "Classify this airline chatbot question into exactly one of: "
                "policy, general_qa, out_of_domain.\n\n"
                "policy     = question about specific airline rules, fees, or policies "
                "(refunds, cancellations, baggage fees, seat upgrades, check-in deadlines, "
                "pet policy, special assistance, loyalty miles, payment methods, meal policy)\n"
                "general_qa = general travel or aviation question not about a specific airline policy "
                "(wifi on planes, in-flight entertainment, airport security rules, boarding process, "
                "travel documents, flight tracking, booking tips, connecting flights, airport arrival)\n"
                "out_of_domain = completely unrelated to airlines or travel\n\n"
                "Reply with ONE word only: policy, general_qa, or out_of_domain."
            ),
            messages=[{"role": "user", "content": text}],
        )
        result = msg.content[0].text.strip().lower().split()[0]
        print(f"[INTENT LLM-Q] '{text[:50]}' -> {result}")
        return result if result in ("policy", "general_qa", "out_of_domain") else "general_qa"
    except Exception:
        return "general_qa"


def _matches(text: str, keywords: list) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def intent_router_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    conv_state = state.get("conv_state", "IDLE")
    input_type = state.get("input_type") or "query"  # set by input_classifier

    # ── Branch: answer → user is responding to bot, route by conv_state ─────
    if input_type == "answer":
        print(f"[INTENT] input_type=answer -> normal")
        return {**state, "intent": "normal", "response_type": "llm"}

    # ── Branch: question → user is asking for information ────────────────────
    if input_type == "question":
        intent = None

        # L1: keyword match for policy
        if _matches(user_input, POLICY_KEYWORDS):
            intent = "policy"
        # L1: customer complaint (separate from policy)
        elif _matches(user_input, CUSTOMER_CONTACT_KEYWORDS):
            intent = "customer_contact"
        # L1: general QA keywords
        elif _matches(user_input, GENERAL_QA_KEYWORDS):
            intent = "general_qa"

        # L2: OOD guard — fires when nothing airline-related in the question
        if intent is None:
            q = user_input.lower()
            if (any(t in q for t in _OOD_TRIGGERS) and
                    not any(a in q for a in _AIRLINE_TERMS)):
                intent = "out_of_domain"

        # L3: embedding similarity
        if intent is None:
            emb_intent, confidence = _embedding_intent(user_input)
            if confidence >= 0.45 and emb_intent in ("policy", "general_qa", "out_of_domain", "customer_contact"):
                intent = emb_intent

        # L4: LLM classification — always consulted for questions if still undecided
        if intent is None:
            intent = _llm_classify_question(user_input)

        # If LLM says out_of_domain, check whether it's actually a greeting first
        if intent == "out_of_domain":
            if _matches(user_input, GREETING_KEYWORDS) or _llm_check_greeting(user_input):
                intent = "greeting"
                print(f"[INTENT] question re-classified as greeting")

        # Final default for questions
        if intent is None:
            intent = "general_qa"

        print(f"[INTENT] input_type=question -> {intent}")
        return {**state, "intent": intent, "response_type": "llm"}

    # ── Branch: query (or unrecognised input_type) → full intent pipeline ────
    intent = None

    # L1: keyword matching across all query-type intents
    if _matches(user_input, TRANSFER_KEYWORDS):
        intent = "transfer"
    elif _matches(user_input, RESTART_KEYWORDS):
        intent = "restart"
    elif _matches(user_input, GREETING_KEYWORDS):        # greetings at ANY conv_state
        intent = "greeting"
    elif _matches(user_input, FIELD_CORRECTION_KEYWORDS):
        intent = "field_correction"
    elif _matches(user_input, CUSTOMER_CONTACT_KEYWORDS):
        intent = "customer_contact"
    elif _matches(user_input, GENERAL_QA_KEYWORDS):
        intent = "general_qa"
    elif _matches(user_input, POLICY_KEYWORDS):
        intent = "policy"

    # L2: OOD guard — fires at any conv_state
    if intent is None:
        q = user_input.lower()
        if (any(t in q for t in _OOD_TRIGGERS) and
                not any(a in q for a in _AIRLINE_TERMS)):
            intent = "out_of_domain"

    # L3: embedding similarity
    if intent is None:
        emb_intent, confidence = _embedding_intent(user_input)
        if confidence >= 0.45 and emb_intent != "normal":
            intent = emb_intent

    # L4: LLM fallback
    if intent is None and len(user_input.strip()) > 3:
        llm_result = _llm_intent(user_input)
        if llm_result not in ("normal", "out_of_domain"):
            intent = llm_result

    # L5: Final fallback — check greeting (LLM), else out_of_domain
    # "normal" is never the last resort; unknown input is either a greeting or OOD.
    if intent is None:
        if _llm_check_greeting(user_input):
            intent = "greeting"
            print(f"[INTENT] fallback greeting detected via LLM")
        else:
            intent = "out_of_domain"
            print(f"[INTENT] fallback: no intent matched -> out_of_domain")

    print(f"[INTENT] input_type=query -> {intent}")
    return {**state, "intent": intent, "response_type": "llm"}
