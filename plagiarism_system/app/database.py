"""Database initialization and session management for plagiarism_system."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DB_PATH = Path(__file__).resolve().parents[1] / "plagiarism_system.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def init_database() -> None:
    """Create all tables if they do not already exist."""
    from . import models  # noqa: F401 - imported for model metadata registration

    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator:
    """Provide transactional SQLAlchemy session scope."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

