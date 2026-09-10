from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.audit_face_gallery import audit_gallery


class FakeGalleryClient:
    def __init__(self):
        self.person_calls = []
        self.face_calls = []

    def list_persons(self, collection, *, limit, cursor):
        self.person_calls.append((collection, limit, cursor))
        if cursor is None:
            return SimpleNamespace(
                persons=[
                    {"id": "p1", "name": "A", "face_count": 1},
                    {"id": "p2", "name": "B", "face_count": 5},
                ],
                next_cursor="next",
            )
        return SimpleNamespace(
            persons=[{"id": "p3", "name": "C", "face_count": 0}],
            next_cursor=None,
        )

    def list_faces(self, collection, person_id, *, limit, cursor):
        self.face_calls.append((collection, person_id, limit, cursor))
        faces = {
            "p1": [{"quality": {"pose": 0.2}}],
            "p2": [
                {"quality": {"pose": 0.8}},
                {"quality": {"pose": 0.9}},
                {"quality": {}},
                {"quality": {"pose": 0.7}},
                {"quality": {"pose": 0.75}},
            ],
            "p3": [],
        }
        return SimpleNamespace(faces=faces[person_id], next_cursor=None)


def test_gallery_audit_uses_person_counts_without_expensive_quality_walk():
    client = FakeGalleryClient()
    report = audit_gallery(client, "all-persons", target_samples=5)

    assert report["person_count"] == 3
    assert report["face_count"] == 6
    assert report["persons_without_faces"] == 1
    assert report["persons_below_target"] == 2
    assert report["sample_coverage_rate"] == pytest.approx(1 / 3)
    assert report["face_count_distribution"] == {"0": 1, "1": 1, "5": 1}
    assert client.face_calls == []


def test_gallery_quality_audit_reports_profile_gaps_and_remediation_people():
    client = FakeGalleryClient()
    report = audit_gallery(
        client,
        "all-persons",
        target_samples=5,
        inspect_quality=True,
        profile_pose_threshold=0.6,
        include_persons=True,
    )

    assert report["persons_with_profile_sample"] == 1
    assert report["persons_needing_profile_sample"] == 2
    assert report["samples_with_pose_quality"] == 5
    assert report["samples_missing_pose_quality"] == 1
    assert [item["person_id"] for item in report["remediation_people"]] == ["p3", "p1", "p2"]
    assert len(client.face_calls) == 3
