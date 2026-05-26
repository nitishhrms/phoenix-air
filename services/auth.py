"""
OTP authentication service + web account login/signup.

OTP flow: generate_otp / verify_otp / send_otp (voice-based auth)
Account flow: create_account / login_account (web UI auth)
"""

import os
import random
import string
import hashlib
from datetime import datetime, timedelta
from db.database import get_conn, create_user_account, get_user_by_phone, get_user_by_email


# ── OTP helpers ───────────────────────────────────────────────────

def generate_otp(phone: str) -> str:
    """Generate a 6-digit OTP, persist to otp_codes, return the code."""
    code = "".join(random.choices(string.digits, k=6))
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    conn = get_conn()
    conn.execute("""
        INSERT INTO otp_codes (phone, code, expires_at, verified)
        VALUES (?, ?, ?, 0)
        ON CONFLICT(phone) DO UPDATE SET
            code       = excluded.code,
            expires_at = excluded.expires_at,
            verified   = 0
    """, (phone, code, expires_at))
    conn.commit()
    conn.close()
    return code


def verify_otp(phone: str, code: str) -> bool:
    """Return True if the code matches and hasn't expired."""
    conn = get_conn()
    row = conn.execute(
        "SELECT code, expires_at, verified FROM otp_codes WHERE phone = ?", (phone,)
    ).fetchone()
    conn.close()
    if not row:
        return False
    if row["verified"]:
        return False
    if datetime.utcnow() > datetime.fromisoformat(row["expires_at"]):
        return False
    if row["code"] != code.strip():
        return False
    conn = get_conn()
    conn.execute("UPDATE otp_codes SET verified = 1 WHERE phone = ?", (phone,))
    conn.commit()
    conn.close()
    return True


def send_otp(phone: str, code: str):
    """
    Send the OTP.
    Priority: Twilio SMS → email (registered email or OTP_FALLBACK_EMAIL) → console.
    Returns: "sms" | "email" | False
    """
    body = f"Your Phoenix Air verification code is: {code}. It expires in 10 minutes."

    # 1 — Try Twilio SMS
    sid   = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_ = os.getenv("TWILIO_FROM_NUMBER", "")
    if sid and token and from_ and not sid.startswith("AC" + "x" * 10):
        try:
            from twilio.rest import Client
            Client(sid, token).messages.create(body=body, from_=from_, to=phone)
            print(f"[OTP SMS SENT] to {phone}")
            return "sms"
        except Exception as e:
            print(f"[OTP SMS ERROR] {e}")

    # 2 — Try email (registered user email → fallback env var)
    to_email = None
    user = get_user_by_phone(phone)
    if user and user.get("email"):
        to_email = user["email"]
    if not to_email:
        to_email = os.getenv("OTP_FALLBACK_EMAIL", "")

    if to_email:
        sent = _send_otp_email(to_email, code)
        if sent:
            return "email"

    # 3 — Dev console fallback
    print(f"\n[OTP DEV MODE] Code for {phone}: {code}\n")
    return False


def _send_otp_email(to: str, code: str) -> bool:
    """Send OTP via Gmail SMTP. Returns True on success."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    host  = os.getenv("SMTP_HOST", "")
    port  = int(os.getenv("SMTP_PORT", "587"))
    user  = os.getenv("SMTP_USER", "")
    pwd   = os.getenv("SMTP_PASS", "")
    from_ = os.getenv("SMTP_FROM", user)

    if not (host and user and pwd):
        return False

    try:
        msg = MIMEMultipart()
        msg["From"]    = from_
        msg["To"]      = to
        msg["Subject"] = "Your Phoenix Air Verification Code"
        body = (
            f"Your Phoenix Air one-time verification code is:\n\n"
            f"  {code}\n\n"
            f"This code expires in 10 minutes. Do not share it with anyone."
        )
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)
        print(f"[OTP EMAIL SENT] to {to}")
        return True
    except Exception as e:
        print(f"[OTP EMAIL ERROR] {e}")
        return False


# ── Web account helpers ───────────────────────────────────────────

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_account(phone: str, first_name: str, last_name: str,
                   email: str, password: str) -> dict:
    """
    Create a new user account. Returns {"success": True, "user_id": N}
    or {"success": False, "error": "..."}.
    """
    if get_user_by_phone(phone):
        return {"success": False, "error": "Phone number already registered."}
    if email and get_user_by_email(email):
        return {"success": False, "error": "Email already registered."}
    password_hash = _hash_password(password)
    return create_user_account(phone, first_name, last_name, email, password_hash)


def login_account(phone_or_email: str, password: str) -> dict | None:
    """
    Verify credentials and return the user row dict, or None if invalid.
    """
    password_hash = _hash_password(password)
    user = get_user_by_phone(phone_or_email) or get_user_by_email(phone_or_email)
    if user and user.get("password_hash") == password_hash:
        return user
    return None
