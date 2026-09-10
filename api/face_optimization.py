"""Quality-aware, bounded aggregation for video face candidates."""

from __future__ import annotations

import math
from collections import defaultdict

from wcm_facerec.config import settings


def _finite(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _box(record):
    box = record.get("face_location") or {}
    values = tuple(_finite(box.get(key), -1.0) for key in ("x", "y", "w", "h"))
    return values if min(values) >= 0 and values[2] > 0 and values[3] > 0 else None


def _iou(a, b):
    if not a or not b:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _track_distance(a, b):
    if not a or not b:
        return math.inf
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    center = math.hypot(ax + aw / 2 - bx - bw / 2, ay + ah / 2 - by - bh / 2)
    scale = max(aw, ah, bw, bh, 0.05)
    size_ratio = max(aw * ah, bw * bh) / max(min(aw * ah, bw * bh), 1e-6)
    return center / scale + (0.35 if size_ratio > 4 else 0.0)


def _quality_weight(record):
    quality = record.get("query_quality") or {}
    # Do not let one heuristic completely erase genuine profile evidence.
    return max(0.25, min(1.0, _finite(quality.get("score"), 0.75)))


def _profile(record):
    quality = record.get("query_quality") or {}
    pose = quality.get("pose")
    return pose is not None and _finite(pose, 1.0) < settings.face_profile_pose_threshold


def observation_is_difficult(observation):
    quality = observation.get("query_quality") or {}
    pose = quality.get("pose")
    sharpness = quality.get("sharpness")
    return (pose is not None and _finite(pose, 1.0) < settings.face_profile_pose_threshold) or (
        sharpness is not None and _finite(sharpness, 1.0) < settings.face_low_sharpness_threshold
    )


def _group_frame_faces(records):
    grouped = {}
    for record in records:
        key = (
            _finite(record.get("frame_time")),
            record.get("face_index"),
            tuple(sorted((record.get("query_face_bbox") or {}).items())),
        )
        grouped.setdefault(key, []).append(record)
    return [
        {
            "time": key[0],
            "box": _box(items[0]),
            "candidates": items,
        }
        for key, items in grouped.items()
    ]


def _build_tracks(records, max_gap):
    tracks = []
    for observation in sorted(_group_frame_faces(records), key=lambda item: item["time"]):
        choices = []
        for index, track in enumerate(tracks):
            gap = observation["time"] - track["last_time"]
            if gap <= 1e-9 or gap > max_gap:
                continue
            distance = _track_distance(track["last_box"], observation["box"])
            overlap = _iou(track["last_box"], observation["box"])
            if overlap >= 0.03 or distance <= 1.75:
                choices.append((distance - overlap, index))
        if choices:
            _, index = min(choices)
            track = tracks[index]
        else:
            track = {"observations": []}
            tracks.append(track)
        track["observations"].append(observation)
        track["last_time"] = observation["time"]
        track["last_box"] = observation["box"]
    return tracks


def _face_sample(record):
    supplied = record.get("_face_sample")
    if supplied:
        return dict(supplied)
    location = record.get("face_location")
    if not location:
        return None
    sample = {
        "time_ms": round(_finite(record.get("frame_time")) * 1000),
        "pts_seconds": _finite(record.get("frame_time")),
        "bbox": location,
    }
    similarity = record.get("similarity")
    if similarity is not None:
        sample["similarity"] = _finite(similarity)
    return sample


def aggregate_face_candidates(records, confirm_similarity, *, max_gap):
    """Return person findings plus diagnostics for one same-shot window."""
    records = list(records)
    if not records:
        return [], {"tracks": 0, "confirmed": 0, "probable": 0, "trigger_times": []}

    # Tests and legacy integrations sometimes return identity-only records.
    # Preserve that contract; production IFS records always carry similarity.
    legacy = [record for record in records if record.get("similarity") is None]
    scored = [record for record in records if record.get("similarity") is not None]
    hits = []
    for record in legacy:
        hit = {
            "source": "face",
            "category": record.get("category") or "敏感人物",
            "description": record.get("name", "敏感人物"),
            "recognition_status": "confirmed",
            "evidence_count": 1,
        }
        sample = _face_sample(record)
        if sample:
            hit["face_samples"] = [sample]
        hits.append(hit)

    diagnostics = {
        "tracks": 0,
        "confirmed": len(hits),
        "probable": 0,
        "trigger_times": [],
    }
    if not scored:
        return hits, diagnostics

    for track in _build_tracks(scored, max_gap):
        diagnostics["tracks"] += 1
        people = defaultdict(list)
        for observation in track["observations"]:
            # One vote per person per presented frame.
            best_at_time = {}
            for candidate in observation["candidates"]:
                key = candidate.get("person_id") or (
                    candidate.get("category"),
                    candidate.get("name"),
                )
                previous = best_at_time.get(key)
                if previous is None or _finite(candidate.get("similarity")) > _finite(
                    previous.get("similarity")
                ):
                    best_at_time[key] = candidate
            for key, candidate in best_at_time.items():
                people[key].append(candidate)

        ranked = []
        for candidates in people.values():
            best_by_time = sorted(
                candidates,
                key=lambda item: _finite(item.get("similarity")),
                reverse=True,
            )[:3]
            total_weight = sum(_quality_weight(item) for item in best_by_time)
            aggregate = sum(
                _finite(item.get("similarity")) * _quality_weight(item) for item in best_by_time
            ) / max(total_weight, 1e-9)
            ranked.append((aggregate, candidates))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            continue

        aggregate, winners = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = aggregate - runner_up
        best = max(winners, key=lambda item: _finite(item.get("similarity")))
        distinct_times = {round(_finite(item.get("frame_time")), 6) for item in winners}
        profile_face = any(_profile(item) for item in winners)
        high_single = (
            len(distinct_times) == 1
            and not profile_face
            and _finite(best.get("similarity"))
            >= max(confirm_similarity, settings.face_high_similarity)
            and margin >= settings.face_min_candidate_margin
        )
        multi_confirmed = (
            len(distinct_times) >= settings.face_min_confirming_frames
            and aggregate >= confirm_similarity
            and margin >= settings.face_min_candidate_margin
        )
        confirmed = high_single or multi_confirmed
        status = "confirmed" if confirmed else "probable"
        if confirmed:
            diagnostics["confirmed"] += 1
        else:
            diagnostics["probable"] += 1
            diagnostics["trigger_times"].extend(distinct_times)

        samples = []
        for candidate in sorted(winners, key=lambda item: _finite(item.get("frame_time"))):
            sample = _face_sample(candidate)
            if sample and sample not in samples:
                samples.append(sample)
        hit = {
            "source": "face",
            "category": best.get("category") or "敏感人物",
            "description": best.get("name", "敏感人物"),
            "recognition_status": status,
            "aggregate_similarity": round(aggregate, 6),
            "evidence_count": len(distinct_times),
            "runner_up_margin": round(margin, 6),
            "profile_face": profile_face,
        }
        if samples:
            hit["face_samples"] = samples
        # Probable hits at or above the user's threshold remain visible for
        # manual review; weaker internal candidates only drive resampling.
        if confirmed or aggregate >= confirm_similarity:
            hits.append(hit)

    diagnostics["trigger_times"] = sorted(set(diagnostics["trigger_times"]))
    return hits, diagnostics
