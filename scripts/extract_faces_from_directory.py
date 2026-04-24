#!/usr/bin/env python3
"""Extract faces from images in a directory, organized by category."""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wcm_facerec.face_engine import FaceEngine


def extract_faces_from_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    min_face_size: int = 30,
    overwrite: bool = False,
) -> dict:
    """Extract faces from all images in a directory structure.

    Args:
        input_dir: Root directory to scan (preserves subdirectory structure)
        output_dir: Root output directory
        min_face_size: Minimum face dimension to accept
        overwrite: Whether to overwrite existing files

    Returns:
        Dict with statistics: {"categories": {}, "total_images": int, "total_faces": int}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = FaceEngine()
    stats = {"categories": defaultdict(lambda: {"images": 0, "faces": 0}), "total_images": 0, "total_faces": 0}

    # Supported image extensions
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    def get_all_image_files(root: Path) -> list[tuple[Path, str]]:
        """Get all image files with their relative category path."""
        images = []
        for item in root.rglob("*"):
            if item.is_file() and item.suffix.lower() in image_exts:
                # Category is the subdirectory name directly under root
                relative = item.relative_to(root)
                category = relative.parts[0] if len(relative.parts) > 1 else "unknown"
                images.append((item, category))
        return images

    def save_face(face_img: np.ndarray, category: str, original_name: Path, face_idx: int, output_root: Path):
        """Save a cropped face maintaining category structure."""
        cat_dir = output_root / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Use original filename as base, add face index if multiple faces
        if face_idx == 0:
            output_name = original_name.stem + "_face" + original_name.suffix
        else:
            output_name = f"{original_name.stem}_face_{face_idx}{original_name.suffix}"

        output_path = cat_dir / output_name

        if not overwrite and output_path.exists():
            return False

        # Convert to uint8 if needed (DeepFace may return float 0-1 images)
        if face_img.dtype != np.uint8:
            face_img = (face_img * 255).astype(np.uint8)

        # DeepFace returns RGB, cv2.imwrite expects BGR - convert
        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_path), face_bgr)
        return True

    def process_image(image_path: Path, category: str) -> int:
        """Process a single image, extract and save faces. Returns face count."""
        import cv2

        try:
            # Load image as numpy array
            img = cv2.imread(str(image_path))
            if img is None:
                return 0

            # Detect faces using numpy array
            faces = engine.detect_faces(img)

            face_count = 0
            for i, face_data in enumerate(faces):
                face_img = face_data.get("face")
                if face_img is None:
                    continue

                # Check minimum size
                h, w = face_img.shape[:2]
                if min(h, w) < min_face_size:
                    continue

                if save_face(face_img, category, image_path, i, output_dir):
                    face_count += 1

            return face_count
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            return 0

    # Process all images
    images = get_all_image_files(input_dir)
    print(f"Found {len(images)} images in {input_dir}")

    for image_path, category in images:
        stats["categories"][category]["images"] += 1
        stats["total_images"] += 1

        face_count = process_image(image_path, category)
        stats["categories"][category]["faces"] += face_count
        stats["total_faces"] += face_count

        print(f"  {category}/{image_path.name}: {face_count} faces extracted")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract faces from images in a directory")
    parser.add_argument("input_dir", help="Input directory to scan")
    parser.add_argument("output_dir", help="Output directory for cropped faces")
    parser.add_argument("--min-size", type=int, default=30, help="Minimum face dimension (default: 30)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Min face size: {args.min_size}px")
    print("---")

    stats = extract_faces_from_directory(
        args.input_dir,
        args.output_dir,
        min_face_size=args.min_size,
        overwrite=args.overwrite,
    )

    print("---")
    print(f"Total: {stats['total_images']} images, {stats['total_faces']} faces extracted")
    print("\nBy category:")
    for cat, data in sorted(stats["categories"].items()):
        print(f"  {cat}: {data['images']} images, {data['faces']} faces")


if __name__ == "__main__":
    main()
