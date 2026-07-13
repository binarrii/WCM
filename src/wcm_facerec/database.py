"""Database connection and models for pgvector."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker, relationship

from .config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class Person(Base):
    """Person model with basic information."""

    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, index=True)
    occupation = Column(String, nullable=True)
    type_ = Column("type", String, nullable=True)
    remarks = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    faces = relationship("FaceRecord", back_populates="person")

    def __repr__(self) -> str:
        return f"<Person(id={self.id}, name={self.name})>"


class FaceRecord(Base):
    """Face record model with pgvector embedding."""

    __tablename__ = "face_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Foreign key to person
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    person = relationship("Person", back_populates="faces")

    def __repr__(self) -> str:
        return f"<FaceRecord(id={self.id}, name={self.name})>"


class SensitiveWord(Base):
    """Sensitive word model for content moderation."""

    __tablename__ = "sensitive_words"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    word = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        {"schema": None},
    )

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
    """Initialize database."""
    engine = get_engine()
    conn = engine.connect()

    # Create tables
    Base.metadata.create_all(engine)
    conn.close()



