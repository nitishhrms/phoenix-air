"""
Mock payment service.
Always succeeds — no real money involved.
Logs the payment to the SQLite payments table.
"""

import random
import string
from datetime import datetime
from db.database import get_conn


def _gen_txn_id() -> str:
    chars = string.ascii_uppercase + string.digits
    return "TXN-" + "".join(random.choices(chars, k=8))


def mock_charge(amount: float, contact: str, confirmation_number: str = "") -> dict:
    """Simulate a payment charge. Always returns success."""
    txn_id = _gen_txn_id()
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO payments
        (transaction_id, confirmation_number, amount, contact, status, created_at)
        VALUES (?, ?, ?, ?, 'success', ?)
    """, (txn_id, confirmation_number, amount, contact, datetime.utcnow()))
    conn.commit()
    conn.close()
    print(f"[PAYMENT] Mock charge of ${amount:.2f} to {contact} — {txn_id}")
    return {"success": True, "transaction_id": txn_id}
