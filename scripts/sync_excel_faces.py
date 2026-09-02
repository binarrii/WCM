"""Synchronize every resolvable ``libface_*.xls`` image into IFS.

One Excel person maps to one IFS Person per collection.  The first image may
create the Person; every other image is enrolled with ``add_faces`` so all
FaceSamples keep that Person's id.  Both the aggregate collection and the
matching category collection are synchronized.

The command is a dry-run by default.  Existing deficient people are checked
image-by-image with an exact-similarity search before an add is planned, which
makes reruns safe and avoids duplicating historical FaceSamples.

Usage::

    uv run python -m scripts.sync_excel_faces
    uv run python -m scripts.sync_excel_faces --apply
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import xlrd

from scripts.backfill_file_path import (
    DEFAULT_AGGREGATE_COLLECTION,
    DEFAULT_BASE_URL,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
    SHEET_NAME,
    SOURCES,
    ExcelRecord,
    SourceSpec,
    _metadata,
    _text,
    find_aggregate_person,
)
from wcm_facerec.vendor.insightface_server import Client

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXTENSION_PRIORITY = {".jpg": 0, ".jpeg": 1, ".png": 2, ".webp": 3, ".bmp": 4}
DEFAULT_EXACT_SIMILARITY = 0.9999
T = TypeVar("T")


@dataclass(frozen=True)
class ExpectedPerson:
    name: str
    images: tuple[ExcelRecord, ...]

    @property
    def metadata(self) -> dict[str, str]:
        representative = self.images[0]
        return {
            "category": representative.category,
            "type": representative.category,
            "occupation": representative.occupation,
            "remarks": representative.remarks,
            "file_path": representative.file_path,
        }


@dataclass(frozen=True)
class FaceAddPlan:
    collection_id: str
    person_id: str
    name: str
    images: tuple[Path, ...]


class FaceSyncClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout: float = 65,
        exact_similarity: float = DEFAULT_EXACT_SIMILARITY,
    ) -> None:
        self.sdk = Client(base_url=base_url, api_key=api_key or None, timeout=timeout)
        self.exact_similarity = exact_similarity

    def close(self) -> None:
        self.sdk.close()

    def all_persons(self, collection_id: str) -> list[dict[str, Any]]:
        people: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = self.sdk.list_persons(collection_id, limit=100, cursor=cursor)
            batch = [dict(person) for person in (page.persons or [])]
            people.extend(batch)
            cursor = page.next_cursor
            if not batch or not cursor:
                return people

    def exact_face_exists(self, collection_id: str, person_id: str, image: Path) -> bool:
        result = self.sdk.search(
            collection_id,
            image=image,
            limit=20,
            threshold=self.exact_similarity,
        )
        for match in result.matches or []:
            person = match.get("person") or {}
            if str(person.get("id") or "") != person_id:
                continue
            if float(match.get("similarity") or 0.0) >= self.exact_similarity:
                return True
        return False

    def create_person(
        self,
        collection_id: str,
        person: ExpectedPerson,
        *,
        person_id: str | None = None,
        external_id: str | None = None,
    ) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
        result = self.sdk.create_person(
            collection_id,
            images=[Path(image.file_path) for image in person.images],
            person_id=person_id,
            name=person.name,
            external_id=external_id,
            metadata=person.metadata,
        )
        return dict(result.person), len(result.faces or []), list(result.rejected_images or [])

    def add_faces(
        self,
        collection_id: str,
        person_id: str,
        images: tuple[Path, ...],
    ) -> tuple[int, list[dict[str, Any]]]:
        result = self.sdk.add_faces(collection_id, person_id, images=images)
        return len(result.faces or []), list(result.rejected_images or [])


def _normalized_stem(value: str) -> str:
    stem = unicodedata.normalize("NFKC", Path(value).stem)
    return re.sub(r"[\s·•]+", "_", stem).strip("_").casefold()


def _logical_candidate(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    logical_stems = {_normalized_stem(candidate.name) for candidate in candidates}
    if len(logical_stems) != 1:
        return None
    return min(
        candidates,
        key=lambda path: (EXTENSION_PRIORITY.get(path.suffix.lower(), 99), path.name),
    )


def _resolve_image(
    filename: str,
    image_dir: Path,
    *,
    by_stem: dict[str, list[Path]],
    by_normalized_stem: dict[str, list[Path]],
    by_numeric_prefix: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    direct = image_dir / Path(filename).name
    if direct.is_file():
        return direct, "exact"

    stem_match = _logical_candidate(by_stem.get(direct.stem, []))
    if stem_match:
        return stem_match, "same_stem"

    normalized_match = _logical_candidate(
        by_normalized_stem.get(_normalized_stem(direct.name), [])
    )
    if normalized_match:
        return normalized_match, "normalized_stem"

    prefix = direct.stem.split("_", 1)[0]
    if prefix.isdigit():
        prefix_match = _logical_candidate(by_numeric_prefix.get(prefix, []))
        if prefix_match:
            return prefix_match, "numeric_prefix"
        if by_numeric_prefix.get(prefix):
            return None, "ambiguous"
    return None, "missing"


def load_expected_people(
    spec: SourceSpec,
    *,
    excel_dir: Path,
    image_root: Path,
) -> tuple[dict[str, ExpectedPerson], dict[str, Any]]:
    workbook_path = excel_dir / spec.workbook
    workbook = xlrd.open_workbook(str(workbook_path))
    if SHEET_NAME not in workbook.sheet_names():
        raise ValueError(f"{workbook_path}: missing sheet {SHEET_NAME!r}")
    sheet = workbook.sheet_by_name(SHEET_NAME)
    headers = [_text(sheet.cell_value(0, column)) for column in range(sheet.ncols)]
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in headers]
    if missing_columns:
        raise ValueError(f"{workbook_path}: missing columns {missing_columns}")
    columns = {column: headers.index(column) for column in REQUIRED_COLUMNS}

    image_dir = image_root / spec.category
    files = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    by_stem: dict[str, list[Path]] = defaultdict(list)
    by_normalized_stem: dict[str, list[Path]] = defaultdict(list)
    by_numeric_prefix: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_stem[path.stem].append(path)
        by_normalized_stem[_normalized_stem(path.name)].append(path)
        prefix = path.stem.split("_", 1)[0]
        if prefix.isdigit():
            by_numeric_prefix[prefix].append(path)

    grouped: dict[str, dict[Path, ExcelRecord]] = defaultdict(dict)
    resolution_counts: dict[str, int] = defaultdict(int)
    unresolved: list[dict[str, str]] = []
    inconsistent_people: set[str] = set()
    first_metadata: dict[str, tuple[str, str, str]] = {}

    for row in range(1, sheet.nrows):
        filename = Path(_text(sheet.cell_value(row, columns["文件名"]))).name
        name = _text(sheet.cell_value(row, columns["姓名"]))
        if not filename or not name:
            continue
        resolved, method = _resolve_image(
            filename,
            image_dir,
            by_stem=by_stem,
            by_normalized_stem=by_normalized_stem,
            by_numeric_prefix=by_numeric_prefix,
        )
        resolution_counts[method] += 1
        if resolved is None:
            unresolved.append({"name": name, "filename": filename, "reason": method})
            continue

        category = _text(sheet.cell_value(row, columns["类型"])) or spec.category
        metadata = (
            _text(sheet.cell_value(row, columns["职业"])),
            category,
            _text(sheet.cell_value(row, columns["备注"])),
        )
        if name in first_metadata and first_metadata[name] != metadata:
            inconsistent_people.add(name)
        first_metadata.setdefault(name, metadata)
        occupation, category, remarks = first_metadata[name]
        grouped[name].setdefault(
            resolved,
            ExcelRecord(
                name=name,
                filename=filename,
                file_path=str(resolved),
                occupation=occupation,
                category=category,
                remarks=remarks,
            ),
        )

    people = {
        name: ExpectedPerson(name=name, images=tuple(records.values()))
        for name, records in grouped.items()
        if records
    }
    report = {
        "workbook": spec.workbook,
        "collection": spec.collection_id,
        "excel_rows": max(sheet.nrows - 1, 0),
        "expected_people": len(people),
        "expected_faces": sum(len(person.images) for person in people.values()),
        "resolution": dict(sorted(resolution_counts.items())),
        "unresolved_images": unresolved,
        "inconsistent_people": len(inconsistent_people),
    }
    return people, report


def _index_by_name(persons: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in persons:
        indexed[_text(person.get("name"))].append(person)
    return indexed


def _aggregate_for_new_category_person(
    name: str,
    category: str,
    aggregate_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    candidates = aggregate_by_name.get(name, [])
    category_matches = [
        person
        for person in candidates
        if (_metadata(person).get("type") or _metadata(person).get("category")) == category
    ]
    if len(category_matches) == 1:
        return category_matches[0]
    if not category_matches and len(candidates) == 1:
        return candidates[0]
    return None


def _with_read_retry(call: Callable[[], T], retries: int) -> T:
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception:  # noqa: BLE001
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def find_missing_images(
    client: FaceSyncClient,
    collection_id: str,
    person_id: str,
    images: tuple[ExcelRecord, ...],
    *,
    workers: int,
    retries: int,
) -> tuple[Path, ...]:
    paths = tuple(Path(image.file_path) for image in images)
    existing: set[Path] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(
                _with_read_retry,
                lambda path=path: client.exact_face_exists(collection_id, person_id, path),
                retries,
            ): path
            for path in paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            if future.result():
                existing.add(path)
    return tuple(path for path in paths if path not in existing)


def _create_missing_person(
    client: FaceSyncClient,
    person: ExpectedPerson,
    *,
    aggregate_collection: str,
    category_collection: str,
    aggregate_person: dict[str, Any] | None,
    category_person: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], int, int, list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    aggregate_added = 0
    category_added = 0

    if aggregate_person is None and category_person is None:
        aggregate_person, aggregate_added, aggregate_rejected = client.create_person(
            aggregate_collection,
            person,
        )
        rejected.extend(
            {"collection": aggregate_collection, **item} for item in aggregate_rejected
        )
        person_id = str(aggregate_person["id"])
        category_person, category_added, category_rejected = client.create_person(
            category_collection,
            person,
            person_id=person_id,
            external_id=person_id,
        )
        rejected.extend(
            {"collection": category_collection, **item} for item in category_rejected
        )
    elif aggregate_person is None:
        person_id = str(category_person["id"])
        aggregate_person, aggregate_added, aggregate_rejected = client.create_person(
            aggregate_collection,
            person,
            person_id=person_id,
        )
        rejected.extend(
            {"collection": aggregate_collection, **item} for item in aggregate_rejected
        )
    elif category_person is None:
        person_id = str(aggregate_person["id"])
        category_person, category_added, category_rejected = client.create_person(
            category_collection,
            person,
            person_id=person_id,
            external_id=person_id,
        )
        rejected.extend(
            {"collection": category_collection, **item} for item in category_rejected
        )

    assert aggregate_person is not None and category_person is not None
    return aggregate_person, category_person, aggregate_added, category_added, rejected


def synchronize(
    client: FaceSyncClient,
    *,
    aggregate_collection: str,
    excel_dir: Path,
    image_root: Path,
    selected_collections: set[str],
    apply: bool,
    workers: int,
    retries: int,
) -> list[dict[str, Any]]:
    aggregate_people = client.all_persons(aggregate_collection)
    aggregate_by_id = {str(person.get("id")): person for person in aggregate_people}
    aggregate_by_external_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in aggregate_people:
        external_id = str(person.get("external_id") or "")
        if external_id:
            aggregate_by_external_id[external_id].append(person)
    aggregate_by_name = _index_by_name(aggregate_people)

    reports: list[dict[str, Any]] = []
    for spec in SOURCES:
        if selected_collections and spec.collection_id not in selected_collections:
            continue
        expected, report = load_expected_people(
            spec,
            excel_dir=excel_dir,
            image_root=image_root,
        )
        category_people = client.all_persons(spec.collection_id)
        category_by_name = _index_by_name(category_people)
        report.update(
            {
                "database_people_before": len(category_people),
                "database_faces_before": sum(
                    int(person.get("face_count") or 0) for person in category_people
                ),
                "new_people": 0,
                "legacy_pair_id_mismatches": 0,
                "deficient_people": 0,
                "exact_checks": 0,
                "planned_face_adds": 0,
                "added_faces": 0,
                "rejected_images": [],
                "unsafe_count_mismatches": [],
                "ambiguous_database_names": [],
            }
        )
        plans: list[FaceAddPlan] = []

        for index, (name, person) in enumerate(sorted(expected.items()), start=1):
            category_matches = category_by_name.get(name, [])
            if len(category_matches) > 1:
                report["ambiguous_database_names"].append(name)
                continue
            category_person = category_matches[0] if category_matches else None
            if category_person is not None:
                aggregate_person = find_aggregate_person(
                    spec.collection_id,
                    category_person,
                    aggregate_by_id=aggregate_by_id,
                    aggregate_by_external_id=aggregate_by_external_id,
                    aggregate_by_name=aggregate_by_name,
                )
            else:
                aggregate_person = _aggregate_for_new_category_person(
                    name,
                    spec.category,
                    aggregate_by_name,
                )

            missing_collections = int(category_person is None) + int(aggregate_person is None)
            if missing_collections:
                category_was_missing = category_person is None
                aggregate_was_missing = aggregate_person is None
                report["new_people"] += 1
                report["planned_face_adds"] += missing_collections * len(person.images)
                if not apply:
                    continue
                (
                    aggregate_person,
                    category_person,
                    aggregate_added,
                    category_added,
                    rejected,
                ) = _create_missing_person(
                    client,
                    person,
                    aggregate_collection=aggregate_collection,
                    category_collection=spec.collection_id,
                    aggregate_person=aggregate_person,
                    category_person=category_person,
                )
                report["added_faces"] += aggregate_added + category_added
                report["rejected_images"].extend(
                    {"name": name, **item} for item in rejected
                )
                if aggregate_was_missing:
                    aggregate_by_id[str(aggregate_person["id"])] = aggregate_person
                    aggregate_by_name[name].append(aggregate_person)
                if category_was_missing:
                    category_by_name[name].append(category_person)

            assert category_person is not None and aggregate_person is not None
            if str(category_person["id"]) != str(aggregate_person["id"]):
                report["legacy_pair_id_mismatches"] += 1

            for collection_id, database_person in (
                (spec.collection_id, category_person),
                (aggregate_collection, aggregate_person),
            ):
                actual = int(database_person.get("face_count") or 0)
                expected_count = len(person.images)
                if actual >= expected_count:
                    continue
                report["deficient_people"] += 1
                report["exact_checks"] += expected_count
                missing = find_missing_images(
                    client,
                    collection_id,
                    str(database_person["id"]),
                    person.images,
                    workers=workers,
                    retries=retries,
                )
                deficit = expected_count - actual
                if len(missing) != deficit:
                    report["unsafe_count_mismatches"].append(
                        {
                            "collection": collection_id,
                            "person_id": str(database_person["id"]),
                            "name": name,
                            "expected": expected_count,
                            "actual": actual,
                            "count_deficit": deficit,
                            "exact_missing": len(missing),
                        }
                    )
                    continue
                report["planned_face_adds"] += len(missing)
                plans.append(
                    FaceAddPlan(
                        collection_id=collection_id,
                        person_id=str(database_person["id"]),
                        name=name,
                        images=missing,
                    )
                )

            if index % 250 == 0:
                print(f"[AUDIT] {spec.collection_id}: {index}/{len(expected)} people")

        if apply:
            for index, plan in enumerate(plans, start=1):
                try:
                    added, rejected = client.add_faces(
                        plan.collection_id,
                        plan.person_id,
                        plan.images,
                    )
                    report["added_faces"] += added
                    report["rejected_images"].extend(
                        {
                            "collection": plan.collection_id,
                            "person_id": plan.person_id,
                            "name": plan.name,
                            **item,
                        }
                        for item in rejected
                    )
                except Exception as exc:  # noqa: BLE001
                    report.setdefault("apply_errors", []).append(
                        {
                            "collection": plan.collection_id,
                            "person_id": plan.person_id,
                            "name": plan.name,
                            "error": str(exc),
                        }
                    )
                if index % 100 == 0:
                    print(f"[APPLY] {spec.collection_id}: {index}/{len(plans)} plans")

        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes; default is dry-run")
    parser.add_argument(
        "--base-url",
        default=os.getenv("WCM_INSIGHTFACE_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--api-key", default=os.getenv("WCM_INSIGHTFACE_API_KEY", ""))
    parser.add_argument(
        "--aggregate-collection",
        default=os.getenv("WCM_INSIGHTFACE_COLLECTION_ID", DEFAULT_AGGREGATE_COLLECTION),
    )
    parser.add_argument("--excel-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image-root", type=Path, default=Path("/tmp/wcm"))
    parser.add_argument(
        "--collection",
        action="append",
        choices=[source.collection_id for source in SOURCES],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=65)
    parser.add_argument("--exact-similarity", type=float, default=DEFAULT_EXACT_SIMILARITY)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")
    if args.retries < 0 or args.retries > 10:
        parser.error("--retries must be between 0 and 10")
    if not 0.99 <= args.exact_similarity <= 1.0:
        parser.error("--exact-similarity must be between 0.99 and 1.0")

    client = FaceSyncClient(
        args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        exact_similarity=args.exact_similarity,
    )
    try:
        reports = synchronize(
            client,
            aggregate_collection=args.aggregate_collection,
            excel_dir=args.excel_dir,
            image_root=args.image_root,
            selected_collections=set(args.collection or []),
            apply=args.apply,
            workers=args.workers,
            retries=args.retries,
        )
    finally:
        client.close()

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if args.apply:
        errors = sum(len(report.get("apply_errors", [])) for report in reports)
        unsafe = sum(len(report["unsafe_count_mismatches"]) for report in reports)
        rejected = sum(len(report["rejected_images"]) for report in reports)
        unresolved = sum(len(report["unresolved_images"]) for report in reports)
        if errors or unsafe or rejected or unresolved:
            raise SystemExit(
                "sync incomplete: "
                f"errors={errors}, unsafe={unsafe}, rejected={rejected}, "
                f"unresolved={unresolved}"
            )
        print("[APPLY] Face synchronization completed successfully.")
    else:
        print("[DRY RUN] No IFS records were changed. Pass --apply to synchronize faces.")


if __name__ == "__main__":
    main()
