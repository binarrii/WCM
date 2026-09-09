"""Count actual review samples, including successful and reused inputs."""

import logging
from bisect import bisect_left, bisect_right

from .review_results import _seconds, flatten_findings

logger = logging.getLogger(__name__)


class ReviewCoverage:
    def __init__(self):
        self.samples = set()
        self.measured = False

    def add(self, timestamps):
        self.measured = True
        self.samples.update(round(timestamp * 1000) for timestamp in timestamps)

    def summarize(self, results):
        if not self.measured:
            return None
        samples = sorted(self.samples)
        incomplete = [
            item for item in flatten_findings(results) if item.get("review_status") == "incomplete"
        ]
        failed = set()
        for item in incomplete:
            try:
                parts = str(item["timestamp"]).split("~")
                start, end = round(_seconds(parts[0]) * 1000), round(_seconds(parts[-1]) * 1000)
                affected = samples[bisect_left(samples, start) : bisect_right(samples, end)]
                if len(parts) > 2 or not affected:
                    raise ValueError("Unmapped incomplete sample")
            except (KeyError, TypeError, ValueError, OverflowError):
                logger.warning("Review coverage unknown: incomplete timestamp cannot be mapped")
                return None
            failed.update(affected)
        return {
            "total_samples": len(samples),
            "incomplete_samples": len(failed),
            "incomplete_checks": len(incomplete),
            "incomplete_ratio": len(failed) / len(samples) if samples else None,
        }


def completion_status(summary):
    total, incomplete = summary["total_samples"], summary["incomplete_samples"]
    if total == 0:
        return "failed"
    # Compare counts before rounding percentages: exactly 10% passes, 30% fails.
    if incomplete * 10 <= total:
        return "completed"
    if incomplete * 10 >= total * 3:
        return "failed"
    return "partial"


def coverage_message(summary):
    total, incomplete = summary["total_samples"], summary["incomplete_samples"]
    if total == 0:
        return "未取得可审核的采样帧，请检查视频内容。"
    if not incomplete:
        return None
    return (
        f"{incomplete} / {total} 个采样点未完成（{incomplete / total:.2%}），"
        f"共 {summary['incomplete_checks']} 项未审核，请查看结果并人工复核。"
    )
