"""Destructively rebuild all SQLAlchemy-managed tables for Voice-AI.

Run from the project root:
    python scripts/database.py

The script drops tables registered in Base.metadata and recreates them.
It does NOT drop the PostgreSQL database itself or remove unregistered objects.
"""


from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.engine import make_url

# Import every model so every table is registered in Base.metadata.
import app.models  # noqa: F401
from app.core.config import settings
from app.core.database import Base, engine


def reset_database() -> None:
    database_url = settings.AIVEN_POSTGRES_URI
    if not database_url:
        raise RuntimeError("AIVEN_POSTGRES_URI is not configured.")

    database_name = make_url(database_url).database
    if not database_name:
        raise RuntimeError(
            "The configured PostgreSQL URL does not specify a database name."
        )

    print("=" * 72)
    print("WARNING: DESTRUCTIVE DATABASE RESET")
    print(f"Configured database: {database_name}")
    print(f"Registered tables to recreate: {len(Base.metadata.tables)}")
    print("All existing SQLAlchemy-managed tables and their data will be dropped.")
    print("The PostgreSQL database itself and unregistered objects are not dropped.")
    print("=" * 72)

    expected_confirmation = f"RESET {database_name}"
    confirmation = input(
        f'Type exactly "{expected_confirmation}" to continue: '
    ).strip()

    if confirmation != expected_confirmation:
        print("Confirmation did not match. No database changes were made.")
        return

    try:
        with engine.begin() as connection:
            Base.metadata.drop_all(bind=connection, checkfirst=True)
            Base.metadata.create_all(bind=connection, checkfirst=True)
    except Exception:
        print("Database reset failed. Review the database error above.")
        raise

    print("Database schema recreated successfully.")
    print("Created tables:")
    for table_name in sorted(Base.metadata.tables):
        print(f"  - {table_name}")


if __name__ == "__main__":
    try:
        reset_database()
    finally:
        engine.dispose()
