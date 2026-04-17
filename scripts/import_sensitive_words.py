#!/usr/bin/env python3
"""Import sensitive words from txt files into database.

Usage:
    python scripts/import_sensitive_words.py [directory]

Directory should contain txt files where:
    - Filename (without .txt) = category name
    - Each line in file = one sensitive word
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wcm_facerec.database import get_session, SensitiveWord


def import_sensitive_words(directory: str | Path) -> dict:
    """Import sensitive words from directory.

    Args:
        directory: Path to directory containing txt files

    Returns:
        Statistics dict with counts
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    session = get_session()
    stats = {"files": 0, "words": 0, "duplicates": 0, "errors": 0}

    try:
        for txt_file in directory.glob("*.txt"):
            category = txt_file.stem  # filename without extension
            stats["files"] += 1

            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if not word:
                        continue

                    # Check for duplicates within same category
                    exists = session.query(SensitiveWord).filter(
                        SensitiveWord.word == word,
                        SensitiveWord.category == category
                    ).first()

                    if exists:
                        stats["duplicates"] += 1
                        continue

                    try:
                        record = SensitiveWord(word=word, category=category)
                        session.add(record)
                        stats["words"] += 1
                    except Exception as e:
                        stats["errors"] += 1
                        print(f"Error inserting '{word}': {e}")

        session.commit()
        print(f"Import complete: {stats}")

    except Exception as e:
        session.rollback()
        print(f"Import failed: {e}")
        raise
    finally:
        session.close()

    return stats


def main():
    if len(sys.argv) < 2:
        # Default directory
        default_dir = Path.home() / "Downloads/2026智能审核/文本"
        if default_dir.exists():
            directory = default_dir
        else:
            print(f"Usage: python {sys.argv[0]} <directory>")
            sys.exit(1)
    else:
        directory = Path(sys.argv[1])

    print(f"Importing sensitive words from: {directory}")
    import_sensitive_words(directory)


if __name__ == "__main__":
    main()
