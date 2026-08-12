from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "fpl.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

SCHEMA_VERSION = 1


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection(db_path)
    try:
        conn.executescript(schema_sql)
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        if row["v"] is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the FPL Agent SQLite database.")
    parser.add_argument("--init", action="store_true", help="Create the database and apply schema.sql")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Override the database file path")
    args = parser.parse_args()

    if args.init:
        init_db(args.db_path)
        print(f"Initialized database at {args.db_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
