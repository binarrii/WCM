from __future__ import annotations

import pytest

from api.face_optimization import aggregate_face_candidates, observation_is_difficult


def _candidate(
    time: float,
    similarity: float,
    *,
    person_id: str = "person-1",
    name: str = "测试人物",
    pose: float = 0.9,
    x: float = 0.1,
):
    return {
        "frame_time": time,
        "face_index": 0,
        "person_id": person_id,
        "name": name,
        "category": "测试分类",
        "similarity": similarity,
        "face_location": {"x": x, "y": 0.1, "w": 0.2, "h": 0.2},
        "query_face_bbox": {"x": 10, "y": 10, "w": 100, "h": 100},
        "query_quality": {"score": 0.8, "sharpness": 0.7, "pose": pose},
    }


def test_two_consistent_frames_confirm_one_identity():
    hits, diagnostics = aggregate_face_candidates(
        [_candidate(1.0, 0.56), _candidate(2.0, 0.58)],
        0.5,
        max_gap=2.5,
    )

    assert len(hits) == 1
    assert hits[0]["recognition_status"] == "confirmed"
    assert hits[0]["evidence_count"] == 2
    assert hits[0]["aggregate_similarity"] == pytest.approx(0.57)
    assert diagnostics["confirmed"] == 1
    assert diagnostics["trigger_times"] == []


def test_high_confidence_frontal_single_frame_keeps_fast_path():
    hits, diagnostics = aggregate_face_candidates([_candidate(1.0, 0.8)], 0.5, max_gap=2.5)

    assert hits[0]["recognition_status"] == "confirmed"
    assert diagnostics["confirmed"] == 1


def test_profile_single_frame_remains_probable_and_requests_neighbors():
    hits, diagnostics = aggregate_face_candidates(
        [_candidate(1.0, 0.8, pose=0.3)], 0.5, max_gap=2.5
    )

    assert hits[0]["recognition_status"] == "probable"
    assert hits[0]["profile_face"] is True
    assert diagnostics["trigger_times"] == [1.0]


def test_close_runner_up_blocks_confirmation():
    records = [
        _candidate(1.0, 0.63),
        _candidate(1.0, 0.61, person_id="person-2", name="相似人物"),
        _candidate(2.0, 0.62),
        _candidate(2.0, 0.60, person_id="person-2", name="相似人物"),
    ]
    hits, diagnostics = aggregate_face_candidates(records, 0.5, max_gap=2.5)

    assert hits[0]["recognition_status"] == "probable"
    assert hits[0]["runner_up_margin"] == pytest.approx(0.02)
    assert diagnostics["trigger_times"] == [1.0, 2.0]


def test_weak_internal_candidate_is_hidden_but_still_requests_neighbors():
    hits, diagnostics = aggregate_face_candidates([_candidate(1.0, 0.4)], 0.5, max_gap=2.5)

    assert hits == []
    assert diagnostics["probable"] == 1
    assert diagnostics["trigger_times"] == [1.0]


@pytest.mark.parametrize(
    "quality",
    [
        {"pose": 0.3, "sharpness": 0.9},
        {"pose": 0.9, "sharpness": 0.05},
    ],
)
def test_pose_or_sharpness_marks_observation_difficult(quality):
    assert observation_is_difficult({"query_quality": quality}) is True


def test_good_or_missing_quality_does_not_force_resampling():
    assert observation_is_difficult({"query_quality": {"pose": 0.9, "sharpness": 0.9}}) is False
    assert observation_is_difficult({}) is False
