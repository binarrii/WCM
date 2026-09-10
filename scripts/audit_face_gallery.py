#!/usr/bin/env python3
"""Audit enrolled face-sample and profile-pose coverage without downloading images."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wcm_facerec.config import settings
from wcm_facerec.vendor.insightface_server import Client


def _all_persons(client: Client, collection: str):
    cursor = None
    while True:
        page = client.list_persons(collection, limit=100, cursor=cursor)
        yield from page.persons
        if not page.next_cursor or page.next_cursor == cursor:
            return
        cursor = page.next_cursor


def _all_faces(client: Client, collection: str, person_id: str):
    cursor = None
    while True:
        page = client.list_faces(collection, person_id, limit=100, cursor=cursor)
        yield from page.faces
        if not page.next_cursor or page.next_cursor == cursor:
            return
        cursor = page.next_cursor


def audit_gallery(
    client: Client,
    collection: str,
    *,
    target_samples: int,
    inspect_quality: bool = False,
    profile_pose_threshold: float = 0.60,
    include_persons: bool = False,
) -> dict[str, Any]:
    """Return an aggregate collection audit and, optionally, a remediation list."""
    distribution: Counter[int] = Counter()
    people: list[dict[str, Any]] = []
    total_faces = 0
    people_with_profile = 0
    quality_samples = 0
    missing_pose_samples = 0

    for person in _all_persons(client, collection):
        face_count = max(0, int(person.get("face_count") or 0))
        total_faces += face_count
        distribution[face_count] += 1
        has_profile_sample = None

        if inspect_quality:
            has_profile_sample = False
            person_faces = list(_all_faces(client, collection, str(person.get("id") or "")))
            # Prefer the page contents when the stored face_count is stale.
            if len(person_faces) != face_count:
                total_faces += len(person_faces) - face_count
                distribution[face_count] -= 1
                if distribution[face_count] <= 0:
                    del distribution[face_count]
                face_count = len(person_faces)
                distribution[face_count] += 1
            for face in person_faces:
                pose = (face.get("quality") or {}).get("pose")
                if pose is None:
                    missing_pose_samples += 1
                    continue
                quality_samples += 1
                if float(pose) < profile_pose_threshold:
                    has_profile_sample = True
            if has_profile_sample:
                people_with_profile += 1

        deficit = max(0, target_samples - face_count)
        needs_profile = bool(inspect_quality and not has_profile_sample)
        if include_persons and (deficit or needs_profile):
            people.append(
                {
                    "person_id": person.get("id"),
                    "name": person.get("name"),
                    "external_id": person.get("external_id"),
                    "face_count": face_count,
                    "sample_deficit": deficit,
                    "has_profile_sample": has_profile_sample,
                }
            )

    person_count = sum(distribution.values())
    people_below_target = sum(
        count for sample_count, count in distribution.items() if sample_count < target_samples
    )
    report: dict[str, Any] = {
        "collection": collection,
        "target_samples_per_person": target_samples,
        "person_count": person_count,
        "face_count": total_faces,
        "average_faces_per_person": round(total_faces / person_count, 3) if person_count else 0.0,
        "persons_without_faces": distribution.get(0, 0),
        "persons_below_target": people_below_target,
        "sample_coverage_rate": round((person_count - people_below_target) / person_count, 6)
        if person_count
        else 0.0,
        "face_count_distribution": {
            str(sample_count): distribution[sample_count] for sample_count in sorted(distribution)
        },
        "quality_inspected": inspect_quality,
    }
    if inspect_quality:
        report.update(
            {
                "profile_pose_threshold": profile_pose_threshold,
                "persons_with_profile_sample": people_with_profile,
                "persons_needing_profile_sample": person_count - people_with_profile,
                "profile_coverage_rate": round(people_with_profile / person_count, 6)
                if person_count
                else 0.0,
                "samples_with_pose_quality": quality_samples,
                "samples_missing_pose_quality": missing_pose_samples,
            }
        )
    if include_persons:
        report["remediation_people"] = sorted(
            people,
            key=lambda item: (
                -int(item["sample_deficit"]),
                str(item.get("name") or ""),
                str(item.get("person_id") or ""),
            ),
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=settings.insightface_base_url)
    parser.add_argument("--api-key", default=settings.insightface_api_key)
    parser.add_argument("--collection", default=settings.insightface_collection_id)
    parser.add_argument("--target-samples", type=int, default=settings.face_gallery_target_samples)
    parser.add_argument(
        "--inspect-quality",
        action="store_true",
        help="Inspect each sample's pose score (more API calls).",
    )
    parser.add_argument(
        "--profile-pose-threshold",
        type=float,
        default=settings.face_profile_pose_threshold,
    )
    parser.add_argument(
        "--include-persons",
        action="store_true",
        help="Include IDs and names needing additional samples in the JSON output.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=settings.insightface_timeout_s)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.target_samples < 1:
        raise SystemExit("--target-samples must be at least 1")
    if not 0 <= args.profile_pose_threshold <= 1:
        raise SystemExit("--profile-pose-threshold must be between 0 and 1")

    with Client(
        args.base_url,
        api_key=args.api_key or None,
        timeout=args.timeout,
    ) as client:
        report = audit_gallery(
            client,
            args.collection,
            target_samples=args.target_samples,
            inspect_quality=args.inspect_quality,
            profile_pose_threshold=args.profile_pose_threshold,
            include_persons=args.include_persons,
        )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
