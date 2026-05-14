#!/usr/bin/env python3
"""CLI tool for face detection and visualization.

Usage:
    python main.py <image_path_or_url> [--output OUTPUT]
    python main.py http://example.com/image.jpg
    python main.py /path/to/image.png --output result.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np

from wcm_facerec.face_engine import FaceEngine


def load_image(source: str) -> tuple[np.ndarray, str]:
    """Load image from path or URL.

    Returns (image_array, source_name).
    """
    if source.startswith(("http://", "https://")):
        resp = httpx.get(source, timeout=30.0)
        resp.raise_for_status()
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from URL: {source}")
        return img, source
    else:
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Failed to read image: {source}")
        return img, source


def draw_faces(img: np.ndarray, faces: list[dict], min_area: int) -> np.ndarray:
    """Draw red boxes around detected faces.

    Args:
        img: Original image (BGR)
        faces: List of face dicts from detect_faces
        min_area: Minimum face area to draw

    Returns:
        Image with drawn boxes
    """
    result = img.copy()
    for i, face in enumerate(faces):
        fa = face.get("facial_area", {})
        x = int(fa.get("x", 0))
        y = int(fa.get("y", 0))
        w = int(fa.get("w", 0))
        h = int(fa.get("h", 0))
        area = w * h
        conf = face.get("confidence", 0)

        if area < min_area:
            label = f"#{i+1} too small ({area})"
            color = (128, 128, 128)  # gray for filtered
        else:
            label = f"#{i+1} area={area} conf={conf:.2f}"
            color = (0, 0, 255)  # red

        # Draw rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x, y - text_h - baseline - 4), (x + text_w, y), color, -1)

        # Draw label text
        text_color = (255, 255, 255) if color == (0, 0, 255) else (0, 0, 0)
        cv2.putText(result, label, (x, y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return result


def main():
    parser = argparse.ArgumentParser(description="Detect faces and draw boxes on image")
    parser.add_argument("image", help="Image file path or URL")
    parser.add_argument("--output", "-o", help="Output image path (default: show with cv2)")
    parser.add_argument("--min-area", type=int, default=0, help="Minimum face area to draw box")
    parser.add_argument("--no-show", action="store_true", help="Don't display image (save only)")
    args = parser.parse_args()

    print(f"Loading: {args.image}")
    img, source_name = load_image(args.image)
    print(f"Image shape: {img.shape}")

    engine = FaceEngine()
    print("Detecting faces...")
    faces = engine.detect_faces(img)
    print(f"Detected {len(faces)} face(s)\n")

    MIN_FACE_PIXELS = 64 * 64
    for i, f in enumerate(faces):
        fa = f.get("facial_area", {})
        area = (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)
        conf = f.get("confidence", 0)
        print(f"Face #{i}:")
        print(f"  bbox: x={fa.get('x')}, y={fa.get('y')}, w={fa.get('w')}, h={fa.get('h')}, area={area}")
        print(f"  confidence: {conf:.4f}")
        print(f"  landmarks:")
        print(f"    left_eye:  {fa.get('left_eye')}")
        print(f"    right_eye: {fa.get('right_eye')}")
        print(f"    nose:      {fa.get('nose')}")
        print(f"    mouth_left:  {fa.get('mouth_left')}")
        print(f"    mouth_right: {fa.get('mouth_right')}")
        print()

    result = draw_faces(img, faces, args.min_area or MIN_FACE_PIXELS)

    passed = sum(1 for f in faces if (f.get("facial_area", {}).get("w", 0) or 0) * (f.get("facial_area", {}).get("h", 0) or 0) >= MIN_FACE_PIXELS)
    print(f"Faces passing area filter ({MIN_FACE_PIXELS}): {passed}/{len(faces)}")

    if args.output:
        cv2.imwrite(args.output, result)
        print(f"Saved: {args.output}")
    else:
        cv2.imshow("Face Detection", result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
