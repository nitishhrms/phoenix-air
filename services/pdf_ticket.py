"""
PDF ticket generator + delivery.
Uses reportlab to build a simple boarding-pass-style PDF.
Sends via SMTP email attachment (if configured) or logs to console.
"""

import os
import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime


def generate_ticket(booking: dict) -> bytes:
    """Return a PDF as bytes. Falls back to a plain-text stub if reportlab is missing."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                     alignment=TA_CENTER, textColor=colors.HexColor("#0071E3"))
        sub_style   = ParagraphStyle("sub",   parent=styles["Normal"],
                                     alignment=TA_CENTER, textColor=colors.grey)

        conf   = booking.get("confirmation_number", "N/A")
        name   = booking.get("passenger_name", "Passenger")
        src    = booking.get("src_iata", "")
        dst    = booking.get("dst_iata", "")
        flight = f"{booking.get('airline','')} {booking.get('flight_number','')}"
        dep    = booking.get("departure_time", "")
        arr    = booking.get("arrival_time", "")

        def fmt_dt(iso):
            try:
                dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                return dt.strftime("%b %d, %Y  %I:%M %p")
            except Exception:
                return iso

        elements = [
            Paragraph("Phoenix Air", title_style),
            Paragraph("Booking Confirmation", sub_style),
            Spacer(1, 0.5*cm),
            Table(
                [
                    ["Confirmation", conf],
                    ["Passenger",    name],
                    ["Flight",       flight],
                    ["Route",        f"{src}  →  {dst}"],
                    ["Departure",    fmt_dt(dep)],
                    ["Arrival",      fmt_dt(arr)],
                ],
                colWidths=[5*cm, 11*cm],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F2F7")),
                    ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE",   (0, 0), (-1, -1), 11),
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F9F9FB")]),
                    ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 12),
                ]),
            ),
            Spacer(1, 1*cm),
            Paragraph("Thank you for flying Phoenix Air. Have a great trip!", sub_style),
        ]
        doc.build(elements)
        return buf.getvalue()

    except ImportError:
        # reportlab not installed — return a plain UTF-8 text stub
        lines = [
            "PHOENIX AIR — BOOKING CONFIRMATION",
            "=" * 40,
            f"Confirmation : {booking.get('confirmation_number', 'N/A')}",
            f"Passenger    : {booking.get('passenger_name', '')}",
            f"Flight       : {booking.get('airline', '')} {booking.get('flight_number', '')}",
            f"Route        : {booking.get('src_iata', '')} → {booking.get('dst_iata', '')}",
            f"Departure    : {booking.get('departure_time', '')}",
            f"Arrival      : {booking.get('arrival_time', '')}",
            "=" * 40,
            "Thank you for flying Phoenix Air!",
        ]
        return "\n".join(lines).encode("utf-8")


def send_ticket(contact: str, contact_type: str, pdf_bytes: bytes, booking: dict):
    """Send the PDF ticket via email, or log to console for phone contacts."""
    conf = booking.get("confirmation_number", "N/A")
    name = booking.get("passenger_name", "Passenger")

    if contact_type == "email":
        _send_email(contact, conf, name, pdf_bytes)
    else:
        # For phone contacts — log (real implementation would use MMS or a link)
        print(f"[PDF TICKET] Would send PDF for booking {conf} to {contact} via SMS/MMS.")


def _send_email(to: str, conf: str, name: str, pdf_bytes: bytes):
    host  = os.getenv("SMTP_HOST", "")
    port  = int(os.getenv("SMTP_PORT", "587"))
    user  = os.getenv("SMTP_USER", "")
    pwd   = os.getenv("SMTP_PASS", "")
    from_ = os.getenv("SMTP_FROM", "Phoenix Air <noreply@phoenixair.com>")

    if not (host and user and pwd):
        print(f"[EMAIL TICKET] SMTP not configured. Would send PDF ticket {conf} to {to}.")
        return

    try:
        msg = MIMEMultipart()
        msg["From"]    = from_
        msg["To"]      = to
        msg["Subject"] = f"Your Phoenix Air Ticket — {conf}"

        body = (
            f"Dear {name},\n\n"
            f"Your booking is confirmed! Please find your ticket attached.\n\n"
            f"Confirmation Number: {conf}\n\n"
            "Thank you for flying Phoenix Air!"
        )
        msg.attach(MIMEText(body, "plain"))

        part = MIMEBase("application", "octet-stream")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="ticket_{conf}.pdf"')
        msg.attach(part)

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)
        print(f"[EMAIL TICKET] Sent to {to}")
    except Exception as e:
        print(f"[EMAIL TICKET ERROR] {e}")
