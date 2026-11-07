from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


if not settings.AIVEN_POSTGRES_URI:
    raise ValueError("AIVEN_POSTGRES_URI is missing from environment variables.")

engine = create_engine(settings.AIVEN_POSTGRES_URI)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a database session and roll back on exceptions."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()





def close_database() -> None:
    engine.dispose()
