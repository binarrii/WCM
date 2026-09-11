from copy import deepcopy

import pytest

from wcm_facerec import person_library
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine
from wcm_facerec.person_library import SameNamePeopleError


@pytest.mark.asyncio
async def test_same_name_checks_all_pages_exact_names_and_all_categories(library):
    engine, seed, _ = library
    calls = []

    def list_persons(**kwargs):
        calls.append(kwargs)
        if kwargs["cursor"] is None:
            return [
                {"id": "similar", "name": "高洁明"},
                {"id": "one", "name": "高洁", "type": "A"},
            ], "next"
        return [{"id": "two", "name": " 高洁 ", "type": "B"}], None

    engine._adapter.list_persons = list_persons
    people = await engine.find_people_by_name(" 高洁 ")
    assert [person["id"] for person in people] == ["one", "two"]
    assert [call["cursor"] for call in calls] == [None, "next"]
    assert all(call["search"] == "高洁" and "collection_id" not in call for call in calls)


@pytest.mark.asyncio
async def test_register_rechecks_same_name_before_any_write(library):
    engine, seed, _ = library
    seed("one", "A", [b"first"])
    original = deepcopy(engine._adapter.people)
    engine._adapter.list_persons = lambda **kwargs: ([{"id": "one", "name": "高洁"}], None)
    with pytest.raises(SameNamePeopleError) as error:
        await engine.register_from_image("高洁", b"new", check_name=True)
    assert error.value.people == [{"id": "one", "name": "高洁"}]
    assert engine._adapter.people == original
    assert engine._adapter.calls == []


class MemoryAdapter:
    def __init__(self):
        self.people = {}
        self.faces = {}
        self.calls = []
        self.fail = None
        self.counter = 0

    def key(self, pid, collection_id):
        return (collection_id or "all", pid)

    def get_person(self, pid, *, collection_id=None):
        return deepcopy(self.people.get(self.key(pid, collection_id)))

    def find_person_by_external_id(self, pid, *, collection_id):
        return next(
            (
                deepcopy(p)
                for (cid, _), p in self.people.items()
                if cid == collection_id and p.get("external_id") == pid
            ),
            None,
        )

    def person_face_ids(self, pid, *, collection_id=None):
        return list(self.faces[self.key(pid, collection_id)])

    def update_person(self, pid, *, metadata, collection_id=None, **kwargs):
        key = self.key(pid, collection_id)
        self.people[key].update(metadata=deepcopy(metadata), **metadata)
        return self.get_person(pid, collection_id=collection_id)

    def add_person_image(self, pid, data, *, collection_id=None):
        key = self.key(pid, collection_id)
        self.counter += 1
        face_id = f"new-{self.counter}"
        self.faces[key][face_id] = data
        self.people[key]["face_count"] = len(self.faces[key])
        self.calls.append(("add", key, data))
        if self.fail == ("add", key):
            self.fail = None
            raise RuntimeError("lost enrollment response")
        return [face_id]

    def delete_person_image(self, pid, face_id, *, collection_id=None):
        key = self.key(pid, collection_id)
        del self.faces[key][face_id]
        self.people[key]["face_count"] = len(self.faces[key])

    def register_person(
        self, *, name, image_bytes, metadata, person_id, collection_id, external_id=None
    ):
        key = (collection_id, person_id)
        assert key not in self.people
        self.people[key] = {
            "id": person_id,
            "name": name,
            "external_id": external_id,
            "metadata": deepcopy(metadata),
            **metadata,
        }
        self.faces[key] = {}
        ids = self.add_person_image(person_id, image_bytes, collection_id=collection_id)
        return person_id, ids[0]

    def delete_person(self, pid, *, collection_id=None):
        key = self.key(pid, collection_id)
        del self.people[key]
        del self.faces[key]
        self.calls.append(("delete", key))
        if self.fail == ("delete", key):
            self.fail = None
            raise RuntimeError("lost delete response")


@pytest.fixture
def library(monkeypatch, tmp_path):
    monkeypatch.setattr(person_library, "IMAGE_ROOT", tmp_path)
    monkeypatch.setattr(settings, "insightface_collection_id", "all")
    monkeypatch.setattr(settings, "insightface_category_collections", {"A": "a", "B": "b"})
    engine = FaceEngine.__new__(FaceEngine)
    engine._adapter = MemoryAdapter()

    def seed(pid, category, contents, *, mirror_id=None):
        paths = []
        for i, content in enumerate(contents):
            path = tmp_path / f"{pid}-{i}.jpg"
            path.write_bytes(content)
            paths.append(str(path))
        metadata = {
            "category": category,
            "type": category,
            "remarks": f"notes-{pid}",
            "file_path": paths[0],
            "image_paths": paths,
            "custom": "keep-me",
        }
        for cid, local_id in [("all", pid), (category.lower(), mirror_id or pid)]:
            engine._adapter.register_person(
                name=f"name-{pid}",
                image_bytes=contents[0],
                metadata=metadata,
                person_id=local_id,
                collection_id=cid,
                external_id=pid,
            )
            for content in contents[1:]:
                engine._adapter.add_person_image(local_id, content, collection_id=cid)
        engine._adapter.calls.clear()
        return paths

    return engine, seed, tmp_path


@pytest.mark.asyncio
async def test_append_keeps_identity_gallery_custom_metadata_and_both_indexes(library):
    engine, seed, root = library
    original = seed("target", "A", [b"first", b"second"], mirror_id="legacy")
    result = await engine.add_person_image("target", b"third")
    assert result["added_images"] == 1
    assert result["record"]["id"] == "target"
    assert result["record"]["image_paths"][:2] == original
    assert len(engine._adapter.people) == 2
    for item in engine._adapter.people.values():
        assert item["face_count"] == 3
        assert item["metadata"]["custom"] == "keep-me"
    duplicate = await engine.add_person_image("target", b"third")
    assert duplicate["added_images"] == 0
    assert len(list((root / "uploads").iterdir())) == 1


@pytest.mark.asyncio
async def test_delete_images_rebuilds_both_indexes_and_keeps_one_enrollment(library):
    engine, seed, root = library
    paths = seed("target", "A", [b"first", b"second", b"third"], mirror_id="legacy")

    result = await engine.delete_person_images("target", [paths[0], paths[2]])

    assert result["removed_images"] == 2
    assert result["record"]["id"] == "target"
    assert result["record"]["file_path"] == paths[1]
    assert result["record"]["image_paths"] == [paths[1]]
    for key in [("all", "target"), ("a", "legacy")]:
        person = engine._adapter.people[key]
        assert person["face_count"] == 1
        assert person["file_path"] == paths[1]
        assert person["image_paths"] == [paths[1]]
        assert list(engine._adapter.faces[key].values()) == [b"second"]
    assert all(person_library.Path(path).exists() for path in paths)


@pytest.mark.asyncio
async def test_delete_images_refuses_last_image_without_mutation(library):
    engine, seed, root = library
    paths = seed("target", "A", [b"first", b"second"])
    before = deepcopy(engine._adapter.people)
    before_samples = {key: list(faces.values()) for key, faces in engine._adapter.faces.items()}

    with pytest.raises(ValueError, match="至少需要保留一张"):
        await engine.delete_person_images("target", paths)

    assert engine._adapter.people == before
    assert {
        key: list(faces.values()) for key, faces in engine._adapter.faces.items()
    } == before_samples
    assert engine._adapter.calls == []


@pytest.mark.asyncio
async def test_delete_images_failure_restores_original_galleries(library):
    engine, seed, root = library
    paths = seed("target", "A", [b"first", b"second", b"third"])
    before = deepcopy(engine._adapter.people)
    before_samples = {key: list(faces.values()) for key, faces in engine._adapter.faces.items()}
    engine._adapter.fail = ("add", ("a", "target"))

    with pytest.raises(RuntimeError, match="lost enrollment response"):
        await engine.delete_person_images("target", [paths[1]])

    assert engine._adapter.people == before
    assert {
        key: list(faces.values()) for key, faces in engine._adapter.faces.items()
    } == before_samples
    journal = next((root / ".person-operations").glob("*.json"))
    assert '"status": "rolled_back"' in journal.read_text()


@pytest.mark.asyncio
async def test_merge_cross_category_legacy_ids_all_photos_and_dedup(library):
    engine, seed, root = library
    original = seed("target", "A", [b"first"])
    source = seed("source", "B", [b"first", b"second"], mirror_id="old-source")
    result = await engine.merge_person_records("target", ["source"])
    assert result["merged_ids"] == ["source"]
    assert result["record"]["image_paths"] == [original[0], source[1]]
    assert result["record"]["name"] == "name-target"
    assert result["record"]["remarks"] == "notes-target"
    assert set(engine._adapter.people) == {("all", "target"), ("a", "target")}
    assert all(p["face_count"] == 2 for p in engine._adapter.people.values())
    assert all(person_library.Path(p).exists() for p in source)
    calls = engine._adapter.calls
    assert max(i for i, call in enumerate(calls) if call[0] == "add") < min(
        i for i, call in enumerate(calls) if call[0] == "delete"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [("add", ("a", "target")), ("delete", ("all", "source"))])
async def test_failure_restores_all_source_samples_and_target_even_after_lost_response(
    library, failure
):
    engine, seed, root = library
    seed("target", "A", [b"first"])
    seed("source", "B", [b"second", b"third"])
    before = deepcopy(engine._adapter.people)
    before_samples = {key: list(faces.values()) for key, faces in engine._adapter.faces.items()}
    engine._adapter.fail = failure
    with pytest.raises(RuntimeError, match="lost"):
        await engine.merge_person_records("target", ["source"])
    assert engine._adapter.people == before
    assert {
        key: list(faces.values()) for key, faces in engine._adapter.faces.items()
    } == before_samples
    journal = next((root / ".person-operations").glob("*.json"))
    assert '"status": "rolled_back"' in journal.read_text()


@pytest.mark.asyncio
async def test_missing_original_refuses_merge_without_mutation(library):
    engine, seed, root = library
    seed("target", "A", [b"first"])
    source = seed("source", "B", [b"second"])
    person_library.Path(source[0]).unlink()
    with pytest.raises(ValueError, match="缺失"):
        await engine.merge_person_records("target", ["source"])
    assert engine._adapter.calls == []


@pytest.mark.asyncio
async def test_missing_mirror_rebuilt_with_entire_gallery(library):
    engine, seed, root = library
    seed("target", "A", [b"first", b"second"])
    engine._adapter.delete_person("target", collection_id="a")
    await engine.add_person_image("target", b"third")
    assert engine._adapter.people[("a", "target")]["face_count"] == 3


@pytest.mark.asyncio
async def test_invalid_or_missing_people_do_not_write(library):
    engine, seed, root = library
    seed("target", "A", [b"first"])
    for sources in [["target"], ["other", "other"], []]:
        with pytest.raises(ValueError):
            await engine.merge_person_records("target", sources)
    with pytest.raises(LookupError):
        await engine.merge_person_records("target", ["missing"])
    assert engine._adapter.calls == []
