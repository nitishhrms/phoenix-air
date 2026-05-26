"""
Unified QA node — policy questions and general airline questions.

Pipeline (tried in order, first successful layer wins):

  Policy path:
    L1 (rule-based)  : keyword match → pre-authored short answer, no LLM
    L2 (embedding)   : TF-IDF cosine retrieval → LLM grounded to policy doc

  General path (when no policy doc found):
    L1 (rule-based)  : keyword match → pre-authored general answer, no LLM
    L2 (embedding)   : dual cosine retrieval on general knowledge docs → LLM grounded
    L3 (web search)  : DuckDuckGo → LLM grounded to search results
    L4 (LLM fallback): answer from model knowledge
    L5 (escalate)    : set intent=customer_contact if LLM also fails

  OOD path:
    Fixed hardcoded decline — no LLM at all
"""

from langchain_core.messages import AIMessage
from services.rag import (
    query_policy_rule_based, query_general_rule_based,
    query_policy_with_score, query_general_with_score,
)
from services.llm import chat_response
from services.hallucination_guard import check_response
import services.web_search as web_search

_FALLBACK_SUPPORT = (
    "For that specific information please contact our support team: "
    "support@phoenixair.com or 1-800-749-2475 (24/7)."
)

_FIXED_OOD = (
    "I specialise in flight booking and Phoenix Air policies — "
    "I'm not able to help with that. "
    "Feel free to ask about flights, baggage, cancellations, or anything else travel-related!"
)

# Terms that confirm a question is airline / travel related
_AIRLINE_TERMS = [
    "flight", "fly", "flying", "airline", "airport", "plane", "aircraft", "travel",
    "trip", "ticket", "booking", "book", "seat", "bag", "baggage", "luggage",
    "check in", "checkin", "board", "boarding", "gate", "depart", "departure",
    "arrive", "arrival", "destination", "route", "passenger", "cabin",
    "meal", "wifi", "wi-fi", "loyalty", "miles", "points", "pet", "infant",
    "upgrade", "refund", "cancel", "policy", "fee", "cost", "price",
    "phoenix air", "carry on", "overhead", "layover", "stopover", "nonstop",
    "international", "domestic", "visa", "passport", "terminal",
    # general airline topics
    "entertainment", "movie", "movies", "screen", "tv", "television", "streaming",
    "connect", "connecting", "connection", "transit", "transfer",
    "track", "tracking", "track my", "status",
    "security", "tsa", "liquid", "customs", "immigration",
    "early", "how early", "arrive at the airport",
    "insurance", "lounge", "delay", "delayed",
    "cheap", "cheapest", "book early", "best price",
]


def _is_airline_related(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in _AIRLINE_TERMS)


def rag_policy_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    messages   = state.get("messages", [])
    language   = state.get("language", "en")
    intent     = state.get("intent", "policy")

    # When intent is already general_qa (set by intent_router), skip Policy L1/L2 entirely.
    # Running general questions through 14 policy docs causes false TF-IDF matches
    # (e.g. "is there wifi" matching meal_policy via structural similarity).
    if intent != "general_qa":
        # ── Policy L1: rule-based keyword match → hardcoded short answer ──
        rule_answer = query_policy_rule_based(user_input)
        if rule_answer:
            print("[HARDCODED] rag_policy: rule-based policy answer")
            response = rule_answer + " Is there anything else I can help you with?"
            msg = AIMessage(content=response)
            return {
                **state,
                "intent":        "normal",
                "response":      response,
                "response_type": "hardcoded",
                "messages":      messages + [msg],
            }

        # ── Policy L2: embedding retrieval → LLM grounded to policy doc ───
        doc_text, score = query_policy_with_score(user_input)

        if doc_text:
            ctx      = {"policy_document": doc_text}
            task     = (
                "Answer the passenger's question using ONLY the policy document provided in context. "
                "Be concise, friendly, and accurate. "
                "After answering, invite them to continue with their booking."
            )
            response = chat_response(task, ctx, user_input, max_tokens=160, language=language) or ""
            passed, _ = check_response(user_input, response, context=doc_text)
            if passed and response:
                msg = AIMessage(content=response)
                return {**state, "intent": "normal", "response": response, "messages": messages + [msg]}
            print(f"[RAG] Policy L2 guard failed (score={score:.3f}) — falling through to general path")

    # ── No policy doc found — general or OOD path ─────────────────────

    if not _is_airline_related(user_input):
        # ── OOD: fixed prompt, no LLM ─────────────────────────────────
        print("[HARDCODED] rag_policy: out-of-domain fixed response")
        msg = AIMessage(content=_FIXED_OOD)
        return {
            **state,
            "intent":        "normal",
            "response":      _FIXED_OOD,
            "response_type": "hardcoded",
            "messages":      messages + [msg],
        }

    # ── General L1: rule-based keyword match → hardcoded general answer
    general_rule = query_general_rule_based(user_input)
    if general_rule:
        print("[HARDCODED] rag_policy: rule-based general answer")
        response = general_rule + " Is there anything else I can help you with?"
        msg = AIMessage(content=response)
        return {
            **state,
            "intent":        "normal",
            "response":      response,
            "response_type": "hardcoded",
            "messages":      messages + [msg],
        }

    # ── General L2: embedding retrieval → LLM grounded to general doc ─
    gen_doc_text, gen_score = query_general_with_score(user_input)
    if gen_doc_text:
        print(f"[GEN EMBED] hit — score {gen_score:.3f}")
        ctx      = {"general_document": gen_doc_text}
        task     = (
            "You are Phoenix Air's assistant. Answer the passenger's question using ONLY the "
            "information in the general document provided in context. "
            "Speak naturally and concisely. Where the document mentions 'airlines generally', "
            "you may say 'most airlines' rather than speaking for Phoenix Air specifically "
            "unless the context is about Phoenix Air. "
            "Invite them to ask anything else."
        )
        fallback = gen_doc_text + " Is there anything else I can help you with?"
        response = chat_response(task, ctx, user_input, max_tokens=160, language=language,
                                 skip_hallucination_guard=True) or fallback
        msg = AIMessage(content=response)
        return {**state, "intent": "normal", "response": response, "messages": messages + [msg]}

    # ── General L3: web search → LLM grounded to results ─────────────
    search_results = web_search.search(user_input)
    if search_results:
        ctx  = {"web_search_results": search_results}
        task = (
            "You are Phoenix Air's assistant. Use the web search results in context to answer "
            "the passenger's question about general airline practices. "
            "Speak about what airlines generally offer — do NOT claim Phoenix Air specifically "
            "has or does not have a feature unless the context confirms it. "
            "Be concise, honest, and invite them to confirm details with our support team if needed."
        )
        response = chat_response(task, ctx, user_input, max_tokens=160, language=language,
                                 skip_hallucination_guard=True) or _FALLBACK_SUPPORT
        msg = AIMessage(content=response)
        return {**state, "intent": "normal", "response": response, "messages": messages + [msg]}

    # ── General L4: LLM from model knowledge ──────────────────────────
    task     = (
        "Answer this airline-related question as Phoenix Air's assistant using your knowledge. "
        "Be concise and helpful. If you truly cannot answer, say so briefly."
    )
    response = chat_response(task, {}, user_input, max_tokens=120, language=language)
    if not response:
        # ── General L5: escalate to customer_contact ──────────────────
        msg = AIMessage(content=_FALLBACK_SUPPORT)
        return {
            **state,
            "intent":   "customer_contact",
            "response": _FALLBACK_SUPPORT,
            "messages": messages + [msg],
        }

    msg = AIMessage(content=response)
    return {**state, "intent": "normal", "response": response, "messages": messages + [msg]}
