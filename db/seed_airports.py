"""Run once: python -m db.seed_airports"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.database import init_db, get_conn
from db.airports_data import AIRPORTS


def seed():
    init_db()
    conn = get_conn()
    conn.executemany(
        "INSERT OR IGNORE INTO airports (iata, city, name, aliases) VALUES (?, ?, ?, ?)",
        AIRPORTS,
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
    conn.close()
    print(f"Seeded {count} airports.")


if __name__ == "__main__":
    seed()
