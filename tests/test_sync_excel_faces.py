from pathlib import Path

from PIL import Image

import scripts.sync_excel_faces as sync_faces
from scripts.backfill_file_path import SOURCES, ExcelRecord
from scripts.sync_excel_faces import (
    ExpectedPerson,
    _create_missing_person,
    _logical_candidate,
    _normalized_stem,
    _resolve_image,
    _upload_image,
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


def test_name_prefix_matches_when_source_hash_changed(tmp_path):
    image = tmp_path / "韩志远_227944e64555eccd2308d830c4b86248.jpg"
    image.touch()

    resolved, method = _resolve_image(
        "韩志远_8cb2543f9119dcef9e50411bfedb1dbc.jpg",
        tmp_path,
        by_stem={image.stem: [image]},
        by_normalized_stem={_normalized_stem(image.name): [image]},
        by_name_prefix={_normalized_stem("韩志远"): [image]},
        by_numeric_prefix={},
    )

    assert resolved == image
    assert method == "name_prefix"


def test_upload_image_normalizes_mislabelled_format(tmp_path):
    path = tmp_path / "actually-a-gif.jpg"
    Image.new("RGB", (20, 20), "white").save(path, format="GIF")

    upload = _upload_image(path)

    assert isinstance(upload, bytes)
    assert upload.startswith(b"\xff\xd8\xff")


def test_upload_image_keeps_correctly_labelled_file(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (20, 20), "white").save(path, format="JPEG")

    assert _upload_image(path) == path


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


def test_expected_person_metadata_contains_every_image_path():
    person = ExpectedPerson(
        name="艾宝俊",
        images=(
            ExcelRecord("艾宝俊", "a.jpg", "/tmp/a.jpg", "无", "落马官员", ""),
            ExcelRecord("艾宝俊", "b.jpg", "/tmp/b.jpg", "无", "落马官员", ""),
        ),
    )

    assert person.metadata["file_path"] == "/tmp/a.jpg"
    assert person.metadata["image_paths"] == ["/tmp/a.jpg", "/tmp/b.jpg"]


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


def test_synchronize_continues_when_new_person_has_no_valid_face(monkeypatch, tmp_path):
    person = ExpectedPerson(
        name="无法识别",
        images=(
            ExcelRecord("无法识别", "a.jpg", "/tmp/a.jpg", "", "劣迹艺人", ""),
        ),
    )

    monkeypatch.setattr(
        sync_faces,
        "load_expected_people",
        lambda *_args, **_kwargs: (
            {person.name: person},
            {
                "workbook": "test.xls",
                "collection": "bad-artists",
                "unresolved_images": [],
            },
        ),
    )

    class RejectingClient:
        def all_persons(self, _collection_id):
            return []

        def create_person(self, *_args, **_kwargs):
            raise RuntimeError("no valid face")

    reports = sync_faces.synchronize(
        RejectingClient(),
        aggregate_collection="all-persons",
        excel_dir=tmp_path,
        image_root=tmp_path,
        selected_collections={"bad-artists"},
        apply=True,
        workers=1,
        retries=0,
    )

    assert reports[0]["apply_errors"] == [
        {
            "phase": "create_person",
            "collection": "bad-artists",
            "name": "无法识别",
            "error": "no valid face",
        }
    ]
