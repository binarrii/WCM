"""Database connection and models.

As of the IFS-as-source-of-truth refactor, this module owns only the
``sensitive_words`` table (used by ``scripts/import_sensitive_words.py``).
The former ``Person`` and ``FaceRecord`` tables — and the ``init_db()``
hook in ``api/main.py`` — have been removed; InsightFace Server is now
the canonical store for face/person data.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class SensitiveWord(Base):
    """Sensitive word model for content moderation."""

    __tablename__ = "sensitive_words"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    word = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = ({"schema": None},)

    def __repr__(self) -> str:
        return f"<SensitiveWord(id={self.id}, word={self.word}, category={self.category})>"


# Engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine


def get_session_factory():
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal


def get_session() -> Session:
    """Get a new database session."""
    return get_session_factory()()


def init_db():
    """Create any tables declared on :class:`Base`.

    Kept for backwards compatibility with offline tooling that still calls
    it on first import. The runtime API container no longer wires this into
    its lifespan hook (Postgres is no longer a deploy dependency).
    """
    Base.metadata.create_all(get_engine())
