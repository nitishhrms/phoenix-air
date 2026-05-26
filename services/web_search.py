"""
Web search service — DuckDuckGo with SQLite caching.

Flow:
  1. Check knowledge_cache table (TTL: 24 hours)
  2. Cache hit  → return cached results immediately
  3. Cache miss → search DuckDuckGo → store in DB → return results
"""

from db.database import get_knowledge_cache, save_knowledge_cache

MAX_RESULTS  = 3
SNIPPET_LEN  = 350


def search(query: str) -> str:
    """
    Search DuckDuckGo for airline context.
    Returns a formatted string ready to pass as LLM context.
    Returns empty string on failure.
    """
    normalized = query.lower().strip()

    cached = get_knowledge_cache(normalized)
    if cached:
        print(f"[WEB SEARCH] cache hit for: {normalized[:60]}")
        return cached

    try:
        from ddgs import DDGS
        # Strip question words so DDG gets keyword-style queries
        import re as _re
        keywords = _re.sub(r'\b(what|is|are|do|does|can|how|when|where|who|there|the|a|an|your|my|i|you)\b', '', query.lower())
        keywords = ' '.join(keywords.split())
        search_query = f"phoenix air airline {keywords}" if keywords else f"phoenix air airline {query}"
        with DDGS() as ddgs:
            raw = list(ddgs.text(search_query, max_results=MAX_RESULTS))

        if not raw:
            return ""

        lines = [f'Web search results for: "{query}"\n']
        for i, r in enumerate(raw, 1):
            title   = r.get("title", "").strip()
            body    = r.get("body", "").strip()[:SNIPPET_LEN]
            lines.append(f"{i}. {title}\n   {body}\n")

        formatted = "\n".join(lines).strip()
        save_knowledge_cache(normalized, formatted)
        print(f"[WEB SEARCH] fetched + cached: {normalized[:60]}")
        return formatted

    except Exception as e:
        print(f"[WEB SEARCH] error: {e}")
        return ""
