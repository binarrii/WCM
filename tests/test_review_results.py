from copy import deepcopy

from api.review_results import consolidate_results, flatten_findings


def test_equal_contained_and_point_findings_share_a_record_without_losing_scope():
    rows = [
        {"timestamp": "55~57", "source": "ocr", "category": "复核", "description": "字幕甲"},
        {"timestamp": "55~57", "source": "visual", "category": "复核", "description": "画面"},
        {"timestamp": "56~57", "source": "ocr", "category": "复核", "description": "字幕乙"},
        {"timestamp": 56, "category": "审核未完成", "review_status": "incomplete", "stage": "face"},
    ]
    before = deepcopy(rows)
    grouped = consolidate_results(rows)
    assert len(grouped) == 1 and grouped[0]["timestamp"] == "55~57"
    assert grouped[0]["review_status"] == "incomplete"
    assert list(flatten_findings(grouped)) == rows
    assert consolidate_results(grouped) == grouped
    assert rows == before


def test_partial_overlaps_and_gaps_do_not_merge_without_window_evidence():
    rows = [{"timestamp": t} for t in ["1~4", "3~5", "6~7", "9~10"]]
    assert consolidate_results(rows) == rows


def test_nested_face_samples_preserve_exact_pts_and_duplicate_records_are_removed():
    face = {
        "timestamp": "2~3",
        "source": "face",
        "description": "甲",
        "face_samples": [
            {
                "time_ms": 2233,
                "pts_seconds": 2 + 7 / 30,
                "duration_seconds": 1 / 30,
                "bbox": {"x": 0.2, "y": 0.1, "w": 0.2, "h": 0.3},
            }
        ],
    }
    grouped = consolidate_results([face, {"timestamp": "1~5", "description": "画面"}, face])
    assert len(grouped) == 1 and len(grouped[0]["findings"]) == 2
    assert grouped[0]["findings"][1] == face


def test_invalid_legacy_interval_is_retained_with_warning(caplog):
    rows = [{"timestamp": "bad", "description": "保留内容"}]
    assert consolidate_results(rows) == rows
    assert "invalid interval" in caplog.text
