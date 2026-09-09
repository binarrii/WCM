import json
from unittest.mock import MagicMock

import pytest

from api import review_task_store
from api.review_coverage import ReviewCoverage, completion_status, coverage_message
from api.review_results import consolidate_results


def gap(time, stage="ocr"):
    return {"timestamp": time, "stage": stage, "review_status": "incomplete"}


@pytest.mark.parametrize(
    "bad,total,status",
    [
        (0, 100, "completed"),
        (1, 10, "completed"),
        (11, 100, "partial"),
        (29, 100, "partial"),
        (3, 10, "failed"),
        (10, 10, "failed"),
        (1001, 10000, "partial"),
        (2999, 10000, "partial"),
        (0, 0, "failed"),
    ],
)
def test_thresholds_use_counts_before_display_rounding(bad, total, status):
    summary = {"total_samples": total, "incomplete_samples": bad}
    assert completion_status(summary) == status


def test_successful_samples_are_denominator_and_overlapping_failures_count_once():
    coverage = ReviewCoverage()
    coverage.add([0, 2, 4, 8, 10, 12, 14, 16, 18, 20])
    coverage.add([2])  # Reused input/timestamp must not inflate the denominator.
    rows = [gap("00:00:00.000~00:00:04.000", "visual"), gap(2), gap(2, "face")]
    summary = coverage.summarize(consolidate_results(rows))
    assert summary == {
        "total_samples": 10,
        "incomplete_samples": 3,
        "incomplete_checks": 3,
        "incomplete_ratio": 0.3,
    }
    assert completion_status(summary) == "failed"


def test_missing_sampling_statistics_is_distinct_from_no_decodable_frames():
    coverage = ReviewCoverage()
    assert coverage.summarize([]) is None
    coverage.add([])
    summary = coverage.summarize([])
    assert completion_status(summary) == "failed"
    assert "未取得" in coverage_message(summary)


@pytest.mark.parametrize("time", ["invalid", "NaN", "00:00:20.000", "1~2~3"])
def test_unmapped_error_does_not_become_success_or_abort_task(time, caplog):
    coverage = ReviewCoverage()
    coverage.add([0, 1, 2])
    assert coverage.summarize([gap(time)]) is None
    assert "coverage unknown" in caplog.text


@pytest.mark.parametrize("bad,status", [(1, "completed"), (2, "partial"), (3, "failed")])
def test_all_finished_statuses_persist_results_and_review_statistics(monkeypatch, bad, status):
    coverage = ReviewCoverage()
    coverage.add(range(10))
    results = [gap(i) for i in range(bad)]
    summary = coverage.summarize(results)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    monkeypatch.setattr(review_task_store, "_connect", lambda: connection)
    review_task_store._complete_sync("task", results, summary)
    saved_status, payload, count, message, saved_summary, task = cursor.execute.call_args.args[1]
    assert saved_status == status
    assert json.loads(saved_summary) == summary
    assert json.loads(payload) == results
    assert count == bad and task == "task"
    assert f"{bad} / 10" in message
    assert f"{bad * 10:.2f}%" in message


def test_samples_belong_to_one_review_only():
    first, second = ReviewCoverage(), ReviewCoverage()
    first.add([0, 1])
    second.add([10])
    assert first.summarize([gap(1)])["incomplete_ratio"] == 0.5
    assert second.summarize([])["total_samples"] == 1
    assert second.summarize([])["incomplete_ratio"] == 0


def test_existing_database_adds_nullable_statistics_column(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = None
    monkeypatch.setattr(review_task_store, "_connect", lambda: connection)
    review_task_store._initialize_sync()
    assert "ADD COLUMN review_summary JSON NULL" in cursor.execute.call_args.args[0]
