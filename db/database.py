import sqlite3
import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH", "./airline.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS airports (
            iata     TEXT PRIMARY KEY,
            city     TEXT NOT NULL,
            name     TEXT NOT NULL,
            aliases  TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id  TEXT PRIMARY KEY,
            conv_state  TEXT NOT NULL DEFAULT 'IDLE',
            data        TEXT NOT NULL DEFAULT '{}',
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_profiles (
            phone       TEXT PRIMARY KEY,
            first_name  TEXT,
            last_name   TEXT,
            last_from   TEXT,
            last_to     TEXT,
            call_count  INTEGER DEFAULT 0,
            updated_at  DATETIME
        );

        CREATE TABLE IF NOT EXISTS bookings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            confirmation_number TEXT UNIQUE,
            session_id          TEXT,
            flight_id           TEXT,
            airline             TEXT,
            flight_number       TEXT,
            departure_time      TEXT,
            arrival_time        TEXT,
            src_iata            TEXT,
            dst_iata            TEXT,
            passenger_name      TEXT,
            contact             TEXT,
            contact_type        TEXT,
            created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS otp_codes (
            phone       TEXT PRIMARY KEY,
            code        TEXT NOT NULL,
            expires_at  DATETIME NOT NULL,
            verified    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS payments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id      TEXT UNIQUE,
            confirmation_number TEXT,
            amount              REAL,
            contact             TEXT,
            status              TEXT DEFAULT 'success',
            created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_accounts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            phone         TEXT UNIQUE NOT NULL,
            email         TEXT,
            first_name    TEXT,
            last_name     TEXT,
            password_hash TEXT NOT NULL,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS judge_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            node_name   TEXT,
            user_input  TEXT,
            response    TEXT,
            language    TEXT DEFAULT 'en',
            relevance   INTEGER,
            accuracy    INTEGER,
            helpfulness INTEGER,
            safety      TEXT,
            reason      TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS knowledge_cache (
            query       TEXT PRIMARY KEY,
            results     TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()


# ── Airport helpers ────────────────────────────────────────────────

def resolve_airport(query: str) -> dict | None:
    q = query.strip().lower()
    conn = get_conn()
    c = conn.cursor()

    row = c.execute(
        "SELECT iata, city, name FROM airports WHERE LOWER(iata) = ?", (q,)
    ).fetchone()

    if not row:
        row = c.execute(
            "SELECT iata, city, name FROM airports "
            "WHERE LOWER(city) = ? OR LOWER(name) = ?", (q, q)
        ).fetchone()

    if not row:
        rows = c.execute("SELECT iata, city, name, aliases FROM airports").fetchall()
        for r in rows:
            aliases = (r["aliases"] or "").lower().split(",")
            for alias in aliases:
                a = alias.strip()
                matched = (a == q) if len(a) <= 3 else (q in a or a in q)
                if matched:
                    row = r
                    break
            if row:
                break

    conn.close()
    if row:
        return {"iata": row["iata"], "city": row["city"], "name": row["name"]}
    return None


def suggest_airports(query: str, top_k: int = 3) -> list:
    q = query.lower().strip()
    if not q:
        return []

    conn = get_conn()
    rows = conn.execute("SELECT iata, city, name, aliases FROM airports").fetchall()
    conn.close()

    scored = []
    for r in rows:
        iata    = r["iata"]
        city    = r["city"]
        name    = r["name"]
        aliases = (r["aliases"] or "")
        all_text = f"{iata} {city} {name} {aliases}".lower()

        if q in all_text:
            scored.append((1.0, iata, city))
            continue

        ratio = SequenceMatcher(None, q, city.lower()).ratio()
        if ratio > 0.45:
            scored.append((ratio, iata, city))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"iata": iata, "city": city, "display": f"{city} ({iata})"}
        for _, iata, city in scored[:top_k]
    ]


# ── Session helpers ────────────────────────────────────────────────

def load_session(session_id: str) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row["data"])
    return {}


def save_session(session_id: str, conv_state: str, data: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO sessions (session_id, conv_state, data, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            conv_state = excluded.conv_state,
            data       = excluded.data,
            updated_at = excluded.updated_at
    """, (session_id, conv_state, json.dumps(data, default=str), datetime.utcnow()))
    conn.commit()
    conn.close()


# ── User profile helpers ───────────────────────────────────────────

def load_user_profile(phone: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE phone = ?", (phone,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_user_profile(phone: str, first_name: str, last_name: str,
                         last_from: str, last_to: str):
    conn = get_conn()
    conn.execute("""
        INSERT INTO user_profiles (phone, first_name, last_name, last_from, last_to, call_count, updated_at)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(phone) DO UPDATE SET
            first_name = excluded.first_name,
            last_name  = excluded.last_name,
            last_from  = excluded.last_from,
            last_to    = excluded.last_to,
            call_count = call_count + 1,
            updated_at = excluded.updated_at
    """, (phone, first_name, last_name, last_from, last_to, datetime.utcnow()))
    conn.commit()
    conn.close()


# ── Session cleanup ───────────────────────────────────────────────

def cleanup_session(session_id: str):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# ── Booking helpers ────────────────────────────────────────────────

def save_booking(data: dict):
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO bookings
        (confirmation_number, session_id, flight_id, airline, flight_number,
         departure_time, arrival_time, src_iata, dst_iata,
         passenger_name, contact, contact_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["confirmation_number"], data["session_id"],
        data["flight_id"], data["airline"], data["flight_number"],
        data["departure_time"], data["arrival_time"],
        data["src_iata"], data["dst_iata"],
        data["passenger_name"], data["contact"], data["contact_type"],
    ))
    conn.commit()
    conn.close()


def get_booking_by_confirmation(confirmation: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM bookings WHERE confirmation_number = ?", (confirmation,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── User account helpers ───────────────────────────────────────────

def create_user_account(phone: str, first_name: str, last_name: str,
                         email: str, password_hash: str) -> dict:
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO user_accounts (phone, email, first_name, last_name, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (phone, email, first_name, last_name, password_hash, datetime.utcnow()))
        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return {"success": True, "user_id": user_id}
    except Exception as e:
        conn.close()
        return {"success": False, "error": str(e)}


def get_user_by_phone(phone: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM user_accounts WHERE phone = ?", (phone,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM user_accounts WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Judge log helpers ──────────────────────────────────────────────

def save_judge_log(data: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO judge_logs
        (session_id, node_name, user_input, response, language,
         relevance, accuracy, helpfulness, safety, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("session_id", ""),
        data.get("node_name", ""),
        data.get("user_input", "")[:300],
        data.get("response", "")[:400],
        data.get("language", "en"),
        data.get("relevance"),
        data.get("accuracy"),
        data.get("helpfulness"),
        data.get("safety"),
        data.get("reason", ""),
        datetime.utcnow(),
    ))
    conn.commit()
    conn.close()


def get_knowledge_cache(query: str, ttl_hours: int = 24) -> str | None:
    from datetime import timedelta
    conn = get_conn()
    cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)
    row = conn.execute(
        "SELECT results FROM knowledge_cache WHERE query = ? AND created_at > ?",
        (query.lower().strip()[:200], cutoff),
    ).fetchone()
    conn.close()
    return row["results"] if row else None


def save_knowledge_cache(query: str, results: str):
    conn = get_conn()
    conn.execute("""
        INSERT INTO knowledge_cache (query, results, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(query) DO UPDATE SET
            results    = excluded.results,
            created_at = excluded.created_at
    """, (query.lower().strip()[:200], results, datetime.utcnow()))
    conn.commit()
    conn.close()


def get_recent_judge_logs(limit: int = 50) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM judge_logs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
