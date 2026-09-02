from pathlib import Path

from scripts.backfill_file_path import SOURCES, ExcelRecord
from scripts.sync_excel_faces import (
    ExpectedPerson,
    _create_missing_person,
    _logical_candidate,
    _normalized_stem,
    find_missing_images,
    load_expected_people,
)


def test_logical_candidate_prefers_jpg_for_same_source(tmp_path):
    jpg = tmp_path / "person_hash.jpg"
    webp = tmp_path / "person_hash.webp"

    assert _logical_candidate([webp, jpg]) == jpg


def test_normalized_stem_matches_unicode_name_separators():
    assert _normalized_stem("艾买尔江•阿吾提_hash.png") == _normalized_stem(
        "艾买尔江_阿吾提_hash.jpg"
    )


def test_load_expected_people_keeps_all_images_for_same_person(tmp_path):
    image_dir = tmp_path / "劣迹艺人"
    image_dir.mkdir()
    filenames = [
        "0000000000016560_62327c22981b26ea0d7faabfb4e60ab0.jpg",
        "0000000000016561_81c8cb9ba69bfa7c894081529aa8a5c2.jpg",
    ]
    for filename in filenames:
        (image_dir / filename).write_bytes(b"image")

    people, report = load_expected_people(
        SOURCES[0],
        excel_dir=Path(__file__).resolve().parent.parent,
        image_root=tmp_path,
    )

    assert [Path(image.file_path).name for image in people["吴秀波"].images] == filenames
    assert report["expected_faces"] == 2


def test_missing_person_uses_same_person_id_in_both_collections():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def create_person(
            self,
            collection_id,
            person,
            *,
            person_id=None,
            external_id=None,
        ):
            assigned = person_id or "person-1"
            self.calls.append((collection_id, assigned, external_id))
            return {"id": assigned, "name": person.name, "face_count": 2}, 2, []

    person = ExpectedPerson(
        name="测试",
        images=(
            ExcelRecord("测试", "a.jpg", "/tmp/a.jpg", "", "劣迹艺人", ""),
        ),
    )
    client = FakeClient()

    aggregate, category, *_ = _create_missing_person(
        client,
        person,
        aggregate_collection="all-persons",
        category_collection="bad-artists",
        aggregate_person=None,
        category_person=None,
    )

    assert aggregate["id"] == category["id"] == "person-1"
    assert client.calls == [
        ("all-persons", "person-1", None),
        ("bad-artists", "person-1", "person-1"),
    ]


def test_find_missing_images_checks_the_existing_person_id(tmp_path):
    present = tmp_path / "present.jpg"
    missing = tmp_path / "missing.jpg"
    present.write_bytes(b"present")
    missing.write_bytes(b"missing")
    images = (
        ExcelRecord("测试", present.name, str(present), "", "劣迹艺人", ""),
        ExcelRecord("测试", missing.name, str(missing), "", "劣迹艺人", ""),
    )

    class FakeClient:
        def exact_face_exists(self, collection_id, person_id, image):
            assert collection_id == "bad-artists"
            assert person_id == "person-1"
            return image == present

    result = find_missing_images(
        FakeClient(),
        "bad-artists",
        "person-1",
        images,
        workers=2,
        retries=0,
    )

    assert result == (missing,)
