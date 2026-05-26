"""
Embedding singleton — disabled on free-tier deployment (512MB RAM limit).
All callers (intent_router, rag, resolve_airport) fall back to TF-IDF + LLM when None is returned.
"""


def get_embedder():
    return None
