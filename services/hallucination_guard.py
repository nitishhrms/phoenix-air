"""
Hallucination guard — uses Claude Haiku to verify that a generated
response is grounded in the provided context and not fabricated.
Returns (True, response) on PASS, (False, reason) on FAIL.
If the guard itself errors or times out, it returns (True, response)
so the flow is never blocked by the guard.
"""

import os


def check_response(user_input: str, response: str, context: str = "") -> tuple[bool, str]:
    """
    Ask Claude Haiku to fact-check the response against the context.
    Returns (passed: bool, text: str).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return True, response

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        ctx_block = f"Context:\n{context}\n\n" if context else ""
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            system=(
                "You are a strict fact-checker for an airline assistant. "
                "Given a passenger question, an optional context document, and a response, "
                "reply with EXACTLY 'PASS' if the response is accurate and grounded, "
                "or 'FAIL: <one-line reason>' if it contains fabricated or incorrect information."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{ctx_block}"
                        f"Question: {user_input}\n"
                        f"Response: {response}"
                    ),
                }
            ],
        )
        verdict = msg.content[0].text.strip()
        if verdict.upper().startswith("FAIL"):
            print(f"[HALLUCINATION GUARD] FAIL — {verdict}")
            return False, verdict
        return True, response
    except Exception as exc:
        print(f"[HALLUCINATION GUARD] skipped (error: {exc})")
        return True, response
