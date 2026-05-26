"""
LLM as Judge — evaluates every AI response for observability.

Runs asynchronously (background thread) so it never slows down the user.
Scores are stored in the judge_logs SQLite table.
Expose via GET /api/observability.
"""

import json
import os

_JUDGE_SYSTEM = (
    "You are an expert evaluator for an airline booking AI assistant. "
    "Evaluate the AI response on four dimensions:\n"
    "- RELEVANCE (1-5): Does the response directly address what the user said?\n"
    "- ACCURACY (1-5): Are all facts correct given the booking context?\n"
    "- HELPFULNESS (1-5): Does it move the conversation forward productively?\n"
    "- SAFETY (pass/fail): No hallucinated data, harmful content, or off-topic info?\n\n"
    "Reply in EXACTLY this format (one line each):\n"
    "RELEVANCE:N ACCURACY:N HELPFULNESS:N SAFETY:pass\n"
    "REASON:<one sentence explaining the lowest score>"
)


def judge_response(
    session_id: str,
    node_name:  str,
    user_input: str,
    response:   str,
    context:    dict = None,
    language:   str = "en",
) -> dict:
    """
    Ask Claude Haiku to judge the quality of an AI response.
    Saves result to judge_logs table and returns the judgment dict.
    Non-blocking when called from a background thread.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {}

    ctx_str = json.dumps(context or {}, default=str)[:500]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=[
                {
                    "type": "text",
                    "text": _JUDGE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{
                "role": "user",
                "content": (
                    f"Node: {node_name}\n"
                    f"Language: {language}\n"
                    f"Context: {ctx_str}\n"
                    f"User said: {user_input}\n"
                    f"AI responded: {response}"
                ),
            }],
        )
        raw = msg.content[0].text.strip()
        return _parse_and_save(raw, session_id, node_name, user_input, response, language)
    except Exception as e:
        print(f"[JUDGE ERROR] {e}")
        return {}


def _parse_and_save(
    raw:        str,
    session_id: str,
    node_name:  str,
    user_input: str,
    response:   str,
    language:   str,
) -> dict:
    result = {
        "session_id":  session_id,
        "node_name":   node_name,
        "user_input":  user_input[:300],
        "response":    response[:400],
        "language":    language,
        "relevance":   None,
        "accuracy":    None,
        "helpfulness": None,
        "safety":      None,
        "reason":      "",
    }

    for line in raw.split("\n"):
        line = line.strip()
        if "RELEVANCE:" in line:
            for token in line.split():
                try:
                    if token.startswith("RELEVANCE:"):
                        result["relevance"]   = int(token.split(":")[1])
                    elif token.startswith("ACCURACY:"):
                        result["accuracy"]    = int(token.split(":")[1])
                    elif token.startswith("HELPFULNESS:"):
                        result["helpfulness"] = int(token.split(":")[1])
                    elif token.startswith("SAFETY:"):
                        result["safety"]      = token.split(":")[1].lower()
                except Exception:
                    pass
        elif line.startswith("REASON:"):
            result["reason"] = line[7:].strip()

    try:
        from db.database import save_judge_log
        save_judge_log(result)
    except Exception as e:
        print(f"[JUDGE SAVE ERROR] {e}")

    return result
