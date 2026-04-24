#!/usr/bin/env python3
"""Standalone batch face registration script.

This script registers faces from a local directory without depending on
the project's internal modules. It can be run independently.

Usage:
    python batch_register.py --input-dir /path/to/faces --xls /path/to/libface.xls

Environment variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD - Database connection
    DEEPFACE_MODEL - Model to use (default: VGG-Face)
"""

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker, relationship

try:
    from pgvector.psycopg2 import register_vector
    from pgvector.sqlalchemy import VECTOR
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    print("Warning: pgvector not available")
    from sqlalchemy import JSON as VECTOR  # Fallback type


@dataclass
class Config:
    """Configuration for batch registration."""

    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5433"))
    db_name: str = os.getenv("DB_NAME", "facerec")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "postgres")
    deepface_model: str = os.getenv("DEEPFACE_MODEL", "VGG-Face")
    embedding_dim: int = 4096  # VGG-Face

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


class Base(DeclarativeBase):
    """SQLAlchemy base."""
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
    """Face record model."""
    __tablename__ = "face_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=True)
    file_url = Column(String, nullable=True)
    embedding = Column(VECTOR(4096), nullable=False)  # VGG-Face dimension
    model = Column(String, nullable=False)
    detector = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    face_id = Column(String, nullable=True)
    frame_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=True)
    person = relationship("Person", back_populates="faces")


def get_embedding_dim(model_name: str) -> int:
    """Get embedding dimension for a model."""
    dims = {
        "VGG-Face": 4096,
        "Facenet": 128,
        "Facenet512": 512,
        "OpenFace": 128,
        "DeepFace": 4096,
        "DeepID": 2048,
        "ArcFace": 512,
        "Dlib": 128,
        "SFace": 128,
    }
    return dims.get(model_name, 4096)


def init_db(config: Config):
    """Initialize database with pgvector extension."""
    engine = create_engine(config.database_url, pool_pre_ping=True)
    conn = engine.connect()

    if PGVECTOR_AVAILABLE:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        print("Using pgvector for embeddings")
    else:
        print("Warning: pgvector not available")

    Base.metadata.create_all(engine)
    conn.close()
    return engine


def extract_name_from_path(path: Path) -> str:
    """Extract person name from file/directory path."""
    if path.is_dir():
        return path.name

    if path.parent.name not in [".", path.root]:
        return path.parent.name

    name = path.stem
    for suffix in ["_face", "_aligned", "_crop"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    if "_" in name:
        parts = name.split("_")
        if len(parts) >= 2:
            potential_hash = parts[1]
            if len(potential_hash) >= 8 and any(c.isalnum() for c in potential_hash):
                return parts[0]

    return name


def get_all_images(directory: Path) -> list[Path]:
    """Get all image files from a directory recursively."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(images)


def generate_embedding(image_path: Path, model_name: str) -> tuple[np.ndarray, float]:
    """Detect face, crop it, then generate embedding using DeepFace.

    Returns:
        Tuple of (embedding, confidence)
    """
    from deepface import DeepFace
    from PIL import Image

    # Load image as numpy array
    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img_array = np.array(img)

    # First extract faces to get cropped face
    faces = DeepFace.extract_faces(
        img_path=img_array,
        enforce_detection=False,
        align=True,
    )

    if not faces:
        raise ValueError(f"No face detected in {image_path}")

    # Sort faces by area and take top 3
    def get_face_area(f):
        fa = f.get("facial_area", {})
        return (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)

    sorted_faces = sorted(faces, key=get_face_area, reverse=True)
    top_faces = sorted_faces[:3]

    results = []
    for best_face in top_faces:
        face_img = best_face.get("face")
        confidence = best_face.get("confidence", 0.0)

        if face_img is None:
            continue

        # Generate embedding from cropped face
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=model_name,
            enforce_detection=False,
            align=True,
        )
        results.append((np.array(embedding[0]["embedding"]), float(confidence)))

    if not results:
        raise ValueError(f"Failed to extract face from {image_path}")

    return results


def load_person_info(xls_path: Path) -> dict[str, dict]:
    """Load person info from xls file.

    Returns a dict mapping filename -> person info
    """
    xl = pd.ExcelFile(xls_path)
    df = xl.parse("faceInfo")

    person_map = {}
    for _, row in df.iterrows():
        filename = row["文件名"]
        person_map[filename] = {
            "name": row["姓名"],
            "occupation": row["职业"] if pd.notna(row["职业"]) else None,
            "type": row["类型"] if pd.notna(row["类型"]) else None,
            "remarks": row["备注"] if pd.notna(row["备注"]) else None,
        }

    return person_map


def get_or_create_person(session: Session, person_info: dict) -> Person | None:
    """Get existing person or create new one."""
    if not person_info or not person_info.get("name"):
        return None

    name = person_info["name"]

    # Check if person already exists
    existing = session.query(Person).filter(Person.name == name).first()
    if existing:
        return existing

    # Create new person
    person = Person(
        id=uuid.uuid4(),
        name=name,
        occupation=person_info.get("occupation"),
        type_=person_info.get("type"),
        remarks=person_info.get("remarks"),
    )
    session.add(person)
    return person


def process_directory(
    directory: Path,
    config: Config,
    session: Session,
    person_map: dict[str, dict],
    dry_run: bool = False,
) -> dict:
    """Process all images in a directory."""
    images = get_all_images(directory)
    stats = {
        "total": len(images),
        "success": 0,
        "failed": 0,
        "persons_created": 0,
        "errors": [],
    }

    print(f"Found {len(images)} images in {directory}")
    print(f"Loaded {len(person_map)} person records from xls")
    print()

    for i, image_path in enumerate(images, 1):
        filename = image_path.name
        person_info = person_map.get(filename, {})
        name = person_info.get("name") or extract_name_from_path(image_path)

        print(f"[{i}/{len(images)}] {filename} (name: {name})")

        if dry_run:
            print(f"  [DRY RUN] Would register: {name}")
            stats["success"] += 1
            continue

        try:
            # Get or create person
            person = get_or_create_person(session, person_info)
            if person and person.id not in session.new:
                stats["persons_created"] += 1
                session.flush()  # Ensure person is persisted

            # Detect and crop faces, then generate embeddings for top 3
            face_results = generate_embedding(image_path, config.deepface_model)

            # Register each face
            for embedding, confidence in face_results:
                record = FaceRecord(
                    id=uuid.uuid4(),
                    name=name,
                    file_path=str(image_path),
                    embedding=embedding.tolist(),  # List for VECTOR type
                    model=config.deepface_model,
                    confidence=confidence,
                    person_id=person.id if person else None,
                )
                session.add(record)
            session.commit()
            stats["success"] += len(face_results)

        except Exception as e:
            stats["failed"] += 1
            error_msg = f"{filename}: {str(e)}"
            stats["errors"].append(error_msg)
            print(f"  ERROR: {e}")
            session.rollback()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch register faces from a local directory with person info",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register faces with xls person info
  python batch_register.py --input-dir /data/faces --xls /data/libface.xls

  # Dry run
  python batch_register.py --input-dir /data/faces --xls /data/libface.xls --dry-run

  # With custom database
  python batch_register.py --input-dir /data/faces --xls /data/libface.xls --db-port 5433
        """,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing face images",
    )
    parser.add_argument(
        "--xls",
        type=Path,
        required=True,
        help="Path to libface.xls file with person info",
    )
    parser.add_argument(
        "--db-host",
        default=os.getenv("DB_HOST", "localhost"),
        help="Database host (default: localhost)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.getenv("DB_PORT", "5433")),
        help="Database port (default: 5433)",
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("DB_NAME", "facerec"),
        help="Database name (default: facerec)",
    )
    parser.add_argument(
        "--db-user",
        default=os.getenv("DB_USER", "postgres"),
        help="Database user (default: postgres)",
    )
    parser.add_argument(
        "--db-password",
        default=os.getenv("DB_PASSWORD", "postgres"),
        help="Database password (default: postgres)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPFACE_MODEL", "VGG-Face"),
        choices=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"],
        help="DeepFace model to use (default: VGG-Face)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be registered without actually registering",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the database tables before processing",
    )

    args = parser.parse_args()

    # Build config
    config = Config(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        deepface_model=args.model,
    )
    config.embedding_dim = get_embedding_dim(args.model)

    if not args.input_dir.exists():
        print(f"Error: Directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not args.xls.exists():
        print(f"Error: XLS file does not exist: {args.xls}")
        sys.exit(1)

    # Load person info
    print(f"Loading person info from: {args.xls}")
    person_map = load_person_info(args.xls)
    print(f"Loaded {len(person_map)} person records")
    print()

    print(f"Connecting to database: {config.db_host}:{config.db_port}/{config.db_name}")
    engine = init_db(config)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Register vector type for pgvector
    if PGVECTOR_AVAILABLE:
        raw_conn = session.connection().connection.driver_connection
        register_vector(raw_conn)

    try:
        print(f"Processing directory: {args.input_dir}")
        print(f"Using model: {config.deepface_model} (embedding dim: {config.embedding_dim})")
        print(f"Dry run: {args.dry_run}")
        print()

        stats = process_directory(
            args.input_dir,
            config,
            session,
            person_map,
            dry_run=args.dry_run,
        )

        print()
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Total images: {stats['total']}")
        print(f"Success: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"Persons created: {stats['persons_created']}")

        if stats["errors"]:
            print(f"\nErrors ({len(stats['errors'])}):")
            for error in stats["errors"][:10]:
                print(f"  - {error}")
            if len(stats["errors"]) > 10:
                print(f"  ... and {len(stats['errors']) - 10} more")

        if stats["failed"] > 0:
            sys.exit(1)

    finally:
        session.close()


if __name__ == "__main__":
    main()
