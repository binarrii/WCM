"""Group review intervals without losing the original findings or their scope."""

import logging
import math
from copy import deepcopy

logger = logging.getLogger(__name__)


def flatten_findings(rows):
    for row in rows:
        if isinstance(row.get("findings"), list):
            yield from flatten_findings(row["findings"])
        else:
            yield row


def _seconds(value):
    parts = str(value).split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return float(value)


def consolidate_results(rows):
    """Equal/contained intervals share one record; partial overlaps remain separate.

    The nested findings keep their own timestamps, sources, evidence and face PTS.
    Legacy summary fields remain available to consumers of flat records.
    """
    entries, invalid = [], []
    for original in flatten_findings(rows):
        row = deepcopy(original)
        try:
            parts = str(row["timestamp"]).split("~")
            start, end = _seconds(parts[0]), _seconds(parts[-1])
            if (
                len(parts) > 2
                or not all(map(math.isfinite, (start, end)))
                or end < start
                or start < 0
            ):
                raise ValueError("Invalid interval")
        except (KeyError, TypeError, ValueError):
            logger.warning("Review result has an invalid interval; retaining it unchanged")
            invalid.append(row)
            continue
        entries.append((start, end, row))
    groups, outer = [], None
    for _start, end, row in sorted(entries, key=lambda entry: (entry[0], -entry[1])):
        if outer is None or end > outer["end"]:
            outer = {"end": end, "rows": []}
            groups.append(outer)
        if row not in outer["rows"]:
            outer["rows"].append(row)
    output = []
    for group in groups:
        findings = group["rows"]
        if len(findings) == 1:
            output.append(findings[0])
            continue
        categories = list(dict.fromkeys(row.get("category", "未分类") for row in findings))
        sources = list(dict.fromkeys(row.get("source", "unknown") for row in findings))
        combined = {
            "timestamp": findings[0]["timestamp"],
            "category": categories[0] if len(categories) == 1 else "综合审核",
            "source": sources[0] if len(sources) == 1 else "mixed",
            "description": "\n\n".join(
                dict.fromkeys(
                    f"{row.get('category', '未分类')}：{row.get('description', '')}"
                    for row in findings
                )
            ),
            "findings": findings,
        }
        if any(row.get("review_status") == "incomplete" for row in findings):
            combined["review_status"] = "incomplete"
        output.append(combined)
    return output + invalid
