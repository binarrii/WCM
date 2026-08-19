"""Backfill IFS Person metadata from the ``libface_*.xls`` mapping files.

The historical IFS import enrolled faces correctly, but most Person records
did not retain the source image path. The dashboard can only render a card
thumbnail when ``metadata.file_path`` points at a file under ``/tmp/wcm``.

Each legacy Excel workbook contains the authoritative relationship between an
image filename and its person, category, occupation, and remarks. This script
uses that relationship to update both copies of a historical person:

* the category collection (``bad-artists``, ``political``, or
  ``corrupt-officials``); and
* the aggregate ``all-persons`` collection used by the dashboard's default
  view.

The script is a dry-run by default. Pass ``--apply`` to PATCH IFS. It never
creates or deletes people or image files, and it merges the corrected fields
into the existing metadata so unrelated keys survive.

Usage::

    uv run python scripts/backfill_file_path.py
    uv run python scripts/backfill_file_path.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xlrd

DEFAULT_BASE_URL = "http://10.252.25.251:18097"
DEFAULT_AGGREGATE_COLLECTION = "all-persons"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SourceSpec:
    workbook: str
    collection_id: str
    category: str


@dataclass(frozen=True)
class ExcelRecord:
    name: str
    filename: str
    file_path: str
    occupation: str
    category: str
    remarks: str


@dataclass(frozen=True)
class PatchOperation:
    collection_id: str
    person_id: str
    name: str
    metadata: dict[str, Any]


SOURCES = (
    SourceSpec("libface_ljyr.xls", "bad-artists", "劣迹艺人"),
    SourceSpec("libface_szmg.xls", "political", "时政敏感"),
    SourceSpec("libface_lmgy.xls", "corrupt-officials", "落马官员"),
)

SHEET_NAME = "faceInfo"
REQUIRED_COLUMNS = ("文件名", "姓名", "职业", "类型", "备注")


def _text(value: Any) -> str:
    """Normalize an XLS cell to the string stored in IFS metadata."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_excel_records(
    spec: SourceSpec,
    *,
    excel_dir: Path,
    image_root: Path,
) -> tuple[dict[str, ExcelRecord], dict[str, int], list[str]]:
    """Return one representative existing image per person in a workbook."""
    workbook_path = excel_dir / spec.workbook
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Excel mapping not found: {workbook_path}")

    workbook = xlrd.open_workbook(str(workbook_path))
    if SHEET_NAME not in workbook.sheet_names():
        raise ValueError(f"{workbook_path}: missing sheet {SHEET_NAME!r}")
    sheet = workbook.sheet_by_name(SHEET_NAME)
    headers = [_text(sheet.cell_value(0, column)) for column in range(sheet.ncols)]
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in headers]
    if missing_columns:
        raise ValueError(f"{workbook_path}: missing columns {missing_columns}")
    columns = {column: headers.index(column) for column in REQUIRED_COLUMNS}

    records: dict[str, ExcelRecord] = {}
    people_seen: set[str] = set()
    inconsistent_people: set[str] = set()
    missing_file_people: set[str] = set()
    existing_rows = 0
    image_dir = image_root / spec.category

    for row in range(1, sheet.nrows):
        filename = Path(_text(sheet.cell_value(row, columns["文件名"]))).name
        name = _text(sheet.cell_value(row, columns["姓名"]))
        if not filename or not name:
            continue
        people_seen.add(name)
        category = _text(sheet.cell_value(row, columns["类型"])) or spec.category
        candidate = ExcelRecord(
            name=name,
            filename=filename,
            file_path=str(image_dir / filename),
            occupation=_text(sheet.cell_value(row, columns["职业"])),
            category=category,
            remarks=_text(sheet.cell_value(row, columns["备注"])),
        )

        previous = records.get(name)
        if previous and (
            previous.occupation,
            previous.category,
            previous.remarks,
        ) != (candidate.occupation, candidate.category, candidate.remarks):
            inconsistent_people.add(name)

        if Path(candidate.file_path).is_file():
            existing_rows += 1
            # Keep the first existing image listed for a person. Excel order
            # is stable and the image is only a representative dashboard card.
            records.setdefault(name, candidate)
        else:
            missing_file_people.add(name)

    missing_file_people.difference_update(records)
    stats = {
        "excel_rows": max(sheet.nrows - 1, 0),
        "excel_people": len(people_seen),
        "existing_image_rows": existing_rows,
        "mapped_people": len(records),
        "people_without_images": len(missing_file_people),
        "inconsistent_people": len(inconsistent_people),
    }
    warnings = []
    if missing_file_people:
        warnings.append(f"people without local images: {sorted(missing_file_people)[:10]}")
    if inconsistent_people:
        warnings.append(
            f"people with inconsistent Excel metadata: {sorted(inconsistent_people)[:10]}"
        )
    return records, stats, warnings


class IFSClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def all_persons(self, collection_id: str) -> list[dict[str, Any]]:
        people: list[dict[str, Any]] = []
        cursor: str | None = None
        collection = urllib.parse.quote(collection_id, safe="")
        while True:
            query: dict[str, Any] = {"limit": 100}
            if cursor:
                query["cursor"] = cursor
            data = self._request(
                "GET",
                f"/v1/collections/{collection}/persons?{urllib.parse.urlencode(query)}",
            )
            page = data.get("persons") or []
            people.extend(page)
            cursor = data.get("next_cursor")
            if not page or not cursor:
                return people

    def patch_metadata(
        self,
        collection_id: str,
        person_id: str,
        metadata: dict[str, Any],
    ) -> None:
        collection = urllib.parse.quote(collection_id, safe="")
        person = urllib.parse.quote(person_id, safe="")
        self._request(
            "PATCH",
            f"/v1/collections/{collection}/persons/{person}",
            {"metadata": metadata},
        )


def _metadata(person: dict[str, Any]) -> dict[str, Any]:
    metadata = person.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    return dict(metadata)


def _desired_metadata(person: dict[str, Any], record: ExcelRecord) -> dict[str, Any]:
    metadata = _metadata(person)
    metadata.update(
        {
            "category": record.category,
            "type": record.category,
            "file_path": record.file_path,
        }
    )
    # Some workbooks contain multiple rows for the same display name with
    # different descriptions. Preserve IFS's existing descriptive metadata;
    # Excel fills it only when the historical import left the field blank.
    if not metadata.get("occupation"):
        metadata["occupation"] = record.occupation
    if not metadata.get("remarks"):
        metadata["remarks"] = record.remarks
    return metadata


def _index_by_name(persons: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in persons:
        index[_text(person.get("name"))].append(person)
    return index


def find_aggregate_person(
    collection_id: str,
    category_person: dict[str, Any],
    *,
    aggregate_by_id: dict[str, dict[str, Any]],
    aggregate_by_external_id: dict[str, list[dict[str, Any]]],
    aggregate_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Resolve new and legacy category mirrors to the aggregate person."""
    person_id = str(category_person.get("id") or "")
    for candidate_id in (person_id, f"{collection_id}-{person_id}"):
        if candidate_id in aggregate_by_id:
            return aggregate_by_id[candidate_id]

    external_matches = aggregate_by_external_id.get(person_id, [])
    if len(external_matches) == 1:
        return external_matches[0]

    name = _text(category_person.get("name"))
    name_matches = aggregate_by_name.get(name, [])
    prefixed = [
        person
        for person in name_matches
        if str(person.get("id") or "").startswith(f"{collection_id}-")
    ]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(name_matches) == 1:
        return name_matches[0]
    return None


def _add_operation(
    operations: dict[tuple[str, str], PatchOperation],
    *,
    collection_id: str,
    person: dict[str, Any],
    record: ExcelRecord,
) -> bool:
    desired = _desired_metadata(person, record)
    if desired == _metadata(person):
        return False
    person_id = str(person["id"])
    key = (collection_id, person_id)
    operation = PatchOperation(
        collection_id=collection_id,
        person_id=person_id,
        name=_text(person.get("name")),
        metadata=desired,
    )
    previous = operations.get(key)
    if previous and previous.metadata != operation.metadata:
        raise ValueError(f"conflicting updates for {collection_id}/{person_id}")
    operations[key] = operation
    return True


def build_operations(
    client: IFSClient,
    *,
    aggregate_collection: str,
    excel_dir: Path,
    image_root: Path,
    selected_collections: set[str],
    limit: int,
) -> tuple[list[PatchOperation], list[dict[str, Any]]]:
    aggregate = client.all_persons(aggregate_collection)
    aggregate_by_id = {str(person.get("id")): person for person in aggregate}
    aggregate_by_external_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for person in aggregate:
        external_id = str(person.get("external_id") or "")
        if external_id:
            aggregate_by_external_id[external_id].append(person)
    aggregate_by_name = _index_by_name(aggregate)

    operations: dict[tuple[str, str], PatchOperation] = {}
    reports: list[dict[str, Any]] = []
    for spec in SOURCES:
        if selected_collections and spec.collection_id not in selected_collections:
            continue
        excel_records, excel_stats, warnings = load_excel_records(
            spec,
            excel_dir=excel_dir,
            image_root=image_root,
        )
        category_people = client.all_persons(spec.collection_id)
        if limit:
            category_people = category_people[:limit]

        report: dict[str, Any] = {
            "collection": spec.collection_id,
            "category": spec.category,
            **excel_stats,
            "category_persons": len(category_people),
            "matched_persons": 0,
            "category_updates": 0,
            "aggregate_updates": 0,
            "already_current": 0,
            "missing_excel_mapping": 0,
            "missing_aggregate_person": 0,
            "warnings": warnings,
        }
        missing_excel: list[str] = []
        missing_aggregate: list[str] = []

        for category_person in category_people:
            name = _text(category_person.get("name"))
            record = excel_records.get(name)
            if record is None:
                report["missing_excel_mapping"] += 1
                if len(missing_excel) < 10:
                    missing_excel.append(name)
                continue
            report["matched_persons"] += 1

            category_changed = _add_operation(
                operations,
                collection_id=spec.collection_id,
                person=category_person,
                record=record,
            )
            if category_changed:
                report["category_updates"] += 1

            aggregate_person = find_aggregate_person(
                spec.collection_id,
                category_person,
                aggregate_by_id=aggregate_by_id,
                aggregate_by_external_id=aggregate_by_external_id,
                aggregate_by_name=aggregate_by_name,
            )
            if aggregate_person is None:
                report["missing_aggregate_person"] += 1
                if len(missing_aggregate) < 10:
                    missing_aggregate.append(name)
                continue
            aggregate_changed = _add_operation(
                operations,
                collection_id=aggregate_collection,
                person=aggregate_person,
                record=record,
            )
            if aggregate_changed:
                report["aggregate_updates"] += 1
            if not category_changed and not aggregate_changed:
                report["already_current"] += 1

        if missing_excel:
            report["warnings"].append(f"IFS people absent from Excel: {missing_excel}")
        if missing_aggregate:
            report["warnings"].append(
                f"category people without aggregate mirror: {missing_aggregate}"
            )
        reports.append(report)

    return list(operations.values()), reports


def _patch_with_retry(
    client: IFSClient,
    operation: PatchOperation,
    retries: int,
) -> None:
    for attempt in range(retries + 1):
        try:
            client.patch_metadata(
                operation.collection_id,
                operation.person_id,
                operation.metadata,
            )
            return
        except Exception:  # noqa: BLE001
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 8))


def apply_operations(
    client: IFSClient,
    operations: list[PatchOperation],
    workers: int,
    retries: int,
) -> int:
    failures: list[tuple[PatchOperation, Exception]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_operation = {
            executor.submit(
                _patch_with_retry,
                client,
                operation,
                retries,
            ): operation
            for operation in operations
        }
        for future in as_completed(future_to_operation):
            operation = future_to_operation[future]
            try:
                future.result()
                completed += 1
                if completed % 500 == 0:
                    print(f"[APPLY] completed {completed}/{len(operations)}")
            except Exception as exc:  # noqa: BLE001
                failures.append((operation, exc))

    for operation, exc in failures[:20]:
        print(f"[ERROR] {operation.collection_id}/{operation.person_id} ({operation.name}): {exc}")
    if failures:
        raise RuntimeError(f"{len(failures)} IFS metadata updates failed")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write updates; default is dry-run")
    parser.add_argument(
        "--base-url",
        default=os.getenv("WCM_INSIGHTFACE_BASE_URL", DEFAULT_BASE_URL),
        help="InsightFace Server base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("WCM_INSIGHTFACE_API_KEY", ""),
        help="optional InsightFace Server bearer token",
    )
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
        help="only process a category collection; repeatable",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="debug: first N IFS people per category"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="parallel PATCH requests for --apply"
    )
    parser.add_argument("--retries", type=int, default=3, help="retries per failed PATCH")
    parser.add_argument("--timeout", type=float, default=30, help="HTTP timeout in seconds")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers must be between 1 and 32")
    if args.retries < 0 or args.retries > 10:
        parser.error("--retries must be between 0 and 10")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    client = IFSClient(args.base_url, args.api_key, args.timeout)
    operations, reports = build_operations(
        client,
        aggregate_collection=args.aggregate_collection,
        excel_dir=args.excel_dir,
        image_root=args.image_root,
        selected_collections=set(args.collection or []),
        limit=args.limit,
    )

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print(f"\nPlanned metadata PATCH operations: {len(operations)}")
    if not args.apply:
        print("[DRY RUN] No IFS records were changed. Pass --apply to write.")
        return

    completed = apply_operations(client, operations, args.workers, args.retries)
    print(f"Applied metadata PATCH operations: {completed}")


if __name__ == "__main__":
    main()
