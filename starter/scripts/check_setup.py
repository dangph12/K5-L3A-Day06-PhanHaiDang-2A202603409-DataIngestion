"""Read-only setup check; this is a utility, not the student pipeline solution."""

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
url = os.getenv("DATABASE_URL")
if not url:
    sys.exit("DATABASE_URL missing: copy starter/.env.example to starter/.env first.")
try:
    with (
        psycopg.connect(url, connect_timeout=5) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "SELECT current_database(), current_setting('server_version'), extversion FROM pg_extension WHERE extname = 'vector'"
        )
        row = cursor.fetchone()
        if not row:
            sys.exit(
                "Connected, but pgvector extension is missing. See starter/README.md."
            )
        print(f"Database: {row[0]} | PostgreSQL: {row[1]} | pgvector: {row[2]}")
except psycopg.Error:
    sys.exit(
        "Database connection failed. Check Docker, port and DATABASE_URL in starter/.env."
    )
print("Setup OK. Pipeline functions still need to be implemented.")
