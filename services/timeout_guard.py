"""
Timeout guard — wraps any callable with a wall-clock deadline.
If the function doesn't return within timeout_s, it is retried once.
If the retry also times out, a safe fallback state is returned.
"""

import concurrent.futures
from typing import Callable, Any


def run_with_timeout(
    fn: Callable,
    args: tuple = (),
    kwargs: dict = None,
    timeout_s: float = 15.0,
    retries: int = 1,
    fallback=None,
) -> Any:
    """
    Run fn(*args, **kwargs) with a timeout.
    Retries once on timeout, then returns fallback.
    """
    if kwargs is None:
        kwargs = {}

    for attempt in range(retries + 1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                future.cancel()
                if attempt < retries:
                    continue
                print(f"[TIMEOUT] {fn.__name__} exceeded {timeout_s}s after {retries + 1} attempt(s).")
                return fallback
            except Exception as exc:
                print(f"[ERROR] {fn.__name__} raised: {exc}")
                raise
