from pathlib import Path

from scripts.backfill_file_path import (
    SOURCES,
    ExcelRecord,
    PatchOperation,
    _desired_metadata,
    _patch_with_retry,
    find_aggregate_person,
    load_excel_records,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_load_excel_records_uses_exact_filename_mapping(tmp_path):
    image_dir = tmp_path / "劣迹艺人"
    image_dir.mkdir()
    filename = "0000000000016736_1108ffef0f7bdb665bb2d19b1bfe115f.png"
    (image_dir / filename).write_bytes(b"test")

    records, stats, warnings = load_excel_records(
        SOURCES[0],
        excel_dir=PROJECT_ROOT,
        image_root=tmp_path,
    )

    assert records["MC天佐"].filename == filename
    assert records["MC天佐"].file_path == str(image_dir / filename)
    assert records["MC天佐"].category == "劣迹艺人"
    assert stats["mapped_people"] == 1
    assert warnings


def test_find_aggregate_person_resolves_legacy_prefixed_id():
    category_person = {"id": "p-bad-artists-00001", "name": "MC天佐"}
    aggregate_person = {"id": "bad-artists-p-bad-artists-00001", "name": "MC天佐"}

    result = find_aggregate_person(
        "bad-artists",
        category_person,
        aggregate_by_id={aggregate_person["id"]: aggregate_person},
        aggregate_by_external_id={},
        aggregate_by_name={"MC天佐": [aggregate_person]},
    )

    assert result is aggregate_person


def test_desired_metadata_merges_unrelated_keys():
    person = {"metadata": {"custom": "keep", "category": "old"}}
    record = ExcelRecord(
        name="MC天佐",
        filename="face.png",
        file_path="/tmp/wcm/劣迹艺人/face.png",
        occupation="网红",
        category="劣迹艺人",
        remarks="说明",
    )

    assert _desired_metadata(person, record) == {
        "custom": "keep",
        "category": "劣迹艺人",
        "occupation": "网红",
        "type": "劣迹艺人",
        "remarks": "说明",
        "file_path": "/tmp/wcm/劣迹艺人/face.png",
    }


def test_patch_with_retry_recovers_from_transient_failure(monkeypatch):
    class FakeClient:
        attempts = 0

        def patch_metadata(self, collection_id, person_id, metadata):
            self.attempts += 1
            if self.attempts < 3:
                raise TimeoutError

    monkeypatch.setattr("scripts.backfill_file_path.time.sleep", lambda _seconds: None)
    client = FakeClient()
    operation = PatchOperation("all-persons", "person-1", "测试", {"file_path": "/tmp/a"})

    _patch_with_retry(client, operation, retries=2)

    assert client.attempts == 3
