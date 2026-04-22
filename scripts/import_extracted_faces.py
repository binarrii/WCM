#!/usr/bin/env python3
"""Import extracted faces into the database."""

import os
import sys
import shutil
import hashlib
import uuid
import tempfile
from pathlib import Path
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wcm_facerec.face_engine import FaceEngine
from wcm_facerec.database import FaceRecord, Person, get_session
from wcm_facerec.config import settings


def import_faces_from_directory(
    input_dir: str | Path,
    category_default: str = "其他",
    overwrite: bool = False,
) -> dict:
    """Import extracted faces into the database.

    Args:
        input_dir: Directory with category subdirectories containing face images
        category_default: Default category for faces without a category
        overwrite: Whether to overwrite existing face records

    Returns:
        Dict with statistics
    """
    input_dir = Path(input_dir)
    engine = FaceEngine()
    stats = defaultdict(lambda: {"added": 0, "skipped": 0, "errors": 0})

    # Get all category directories
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Category mapping (directory name -> type/remarks)
    category_map = {
        "人物": {"type": "普通人物", "remarks": ""},
        "劣迹": {"type": "劣迹艺人", "remarks": ""},
        "敏感人物": {"type": "时政敏感", "remarks": ""},
        "落马": {"type": "落马官员", "remarks": ""},
        "邪教人物": {"type": "邪教人物", "remarks": ""},
    }

    def get_or_create_person(name: str, category: str) -> Person:
        """Get or create a person record."""
        session = get_session()
        try:
            # Check if person with same name exists
            person = session.query(Person).filter(Person.name == name).first()
            if person:
                return person

            # Create new person
            cat_info = category_map.get(category, {"type": category_default, "remarks": ""})
            person = Person(
                id=uuid.uuid4(),
                name=name,
                occupation="",
                type_=cat_info["type"],
                remarks=cat_info["remarks"],
            )
            session.add(person)
            session.commit()
            session.refresh(person)
            return person
        finally:
            session.close()

    def extract_name_from_filename(filename: str) -> str:
        """Extract person name from face filename like '马文辉_face.png' or '张三_face_1.png'."""
        name = Path(filename).stem
        # Remove _face suffix
        if "_face" in name:
            name = name.split("_face")[0]
        return name.strip()

    def process_category_dir(category: str, cat_dir: Path) -> None:
        """Process all face images in a category directory."""
        # Get category info
        cat_info = category_map.get(category, {"type": category_default, "remarks": ""})

        for face_file in cat_dir.iterdir():
            if not face_file.is_file():
                continue

            ext = face_file.suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            try:
                name = extract_name_from_filename(face_file.name)
                if not name:
                    print(f"  Warning: Could not extract name from {face_file.name}")
                    stats[category]["skipped"] += 1
                    continue

                # Get or create person
                person = get_or_create_person(name, category)

                # Copy to temp with ASCII path for DeepFace
                temp_path = Path(tempfile.gettempdir()) / f"wcm_import_{uuid.uuid4().hex[:12]}{ext}"
                shutil.copy2(face_file, temp_path)

                try:
                    # Generate embedding
                    embedding = engine.generate_embedding(temp_path)

                    # Register face in database
                    session = get_session()
                    try:
                        # Check if this face already exists (by embedding hash check would be expensive,
                        # so we just insert and let the caller handle duplicates if needed)
                        record = FaceRecord(
                            id=uuid.uuid4(),
                            name=name,
                            file_path=str(face_file),
                            file_url=None,
                            embedding=embedding.tolist(),
                            model=settings.deepface_model,
                            confidence=None,
                            face_id=f"face_{hashlib.md5(str(face_file).encode()).hexdigest()[:8]}",
                            frame_time=None,
                            person_id=person.id,
                        )
                        session.add(record)
                        session.commit()
                        stats[category]["added"] += 1
                        print(f"  {name} ({category}): added")
                    except Exception as e:
                        session.rollback()
                        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                            stats[category]["skipped"] += 1
                            print(f"  {name} ({category}): skipped (duplicate)")
                        else:
                            stats[category]["errors"] += 1
                            print(f"  {name} ({category}): error - {e}")
                    finally:
                        session.close()
                finally:
                    if temp_path.exists():
                        temp_path.unlink()

            except Exception as e:
                stats[category]["errors"] += 1
                print(f"  Error processing {face_file.name}: {e}")

    # Process each category directory
    for item in input_dir.iterdir():
        if not item.is_dir():
            continue

        category = item.name
        print(f"\nProcessing category: {category}")
        process_category_dir(category, item)

    return dict(stats)


def main():
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Import extracted faces into database")
    parser.add_argument("input_dir", help="Directory with extracted faces (category subdirs)")
    parser.add_argument("--category-default", default="其他", help="Default category name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing records")

    args = parser.parse_args()

    print(f"Input: {args.input_dir}")
    print(f"Default category: {args.category_default}")
    print("---")

    stats = import_faces_from_directory(
        args.input_dir,
        category_default=args.category_default,
        overwrite=args.overwrite,
    )

    print("\n--- Summary ---")
    total_added = sum(v["added"] for v in stats.values())
    total_skipped = sum(v["skipped"] for v in stats.values())
    total_errors = sum(v["errors"] for v in stats.values())
    print(f"Total: {total_added} added, {total_skipped} skipped, {total_errors} errors")
    print("\nBy category:")
    for cat, v in sorted(stats.items()):
        print(f"  {cat}: {v['added']} added, {v['skipped']} skipped, {v['errors']} errors")


if __name__ == "__main__":
    main()
