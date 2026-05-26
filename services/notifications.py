import re
import os
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

US_PHONE_RE = re.compile(r'^\+1[2-9]\d{9}$')
PHONE_RE    = re.compile(r'^\+?[\d\s\-().]{7,20}$')
EMAIL_RE    = re.compile(r'^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$')


def detect_contact_type(contact: str) -> str:
    """Return 'phone' or 'email'."""
    c = contact.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if EMAIL_RE.match(contact.strip()):
        return "email"
    if PHONE_RE.match(c):
        return "phone"
    return "email"


def is_us_phone(contact: str) -> bool:
    c = contact.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    return bool(US_PHONE_RE.match(c))


def _twilio_send(to: str, body: str):
    """Send real SMS via Twilio in a background thread."""
    sid   = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_ = os.getenv("TWILIO_FROM_NUMBER", "")
    if not (sid and token and from_):
        print(f"[SMS SKIPPED] Twilio env vars not set. Would send to {to}")
        return
    try:
        from twilio.rest import Client
        Client(sid, token).messages.create(body=body, from_=from_, to=to)
        print(f"[SMS SENT] to {to}")
    except Exception as e:
        print(f"[SMS ERROR] {e}")


def send_confirmation(contact: str, contact_type: str, booking: dict):
    name = booking.get("passenger_name", "Passenger")
    conf = booking.get("confirmation_number", "N/A")
    airline = booking.get("airline", "")
    fn = booking.get("flight_number", "")
    dep = booking.get("departure_time", "")
    arr = booking.get("arrival_time", "")
    src = booking.get("src_iata", "")
    dst = booking.get("dst_iata", "")

    message = (
        f"Hi {name}, your booking is confirmed!\n"
        f"Confirmation: {conf}\n"
        f"Flight: {airline} {fn}\n"
        f"Route: {src} to {dst}\n"
        f"Departs: {dep}  |  Arrives: {arr}\n"
        f"Thank you for flying Phoenix Air!"
    )

    if contact_type == "phone" and is_us_phone(contact):
        threading.Thread(
            target=_twilio_send, args=(contact, message), daemon=True
        ).start()
    else:
        threading.Thread(
            target=_smtp_send, args=(contact, f"Your Phoenix Air Booking {conf}", message), daemon=True
        ).start()


def _smtp_send(to: str, subject: str, body: str):
    host  = os.getenv("SMTP_HOST", "")
    port  = int(os.getenv("SMTP_PORT", "587"))
    user  = os.getenv("SMTP_USER", "")
    pwd   = os.getenv("SMTP_PASS", "")
    from_ = os.getenv("SMTP_FROM", user)

    if not (host and user and pwd):
        print(f"[EMAIL] SMTP not configured. Would send to {to}: {body}")
        return

    try:
        msg = MIMEMultipart()
        msg["From"]    = from_
        msg["To"]      = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)
        print(f"[EMAIL SENT] to {to}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
