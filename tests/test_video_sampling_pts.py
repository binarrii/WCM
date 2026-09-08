"""Sampling follows presentation time, without integer-FPS drift."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from api import handlers, utils


class TimedCapture:
    def __init__(self, times, fps=25):
        self.times, self.fps = times, fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else self.times[self.index - 1] * 1000

    def read(self):
        if self.index == len(self.times):
            return False, None
        self.index += 1
        return True, np.zeros((8, 8, 3), np.uint8)

    def release(self):
        self.released = True


def sample(monkeypatch, times, interval, fps=25):
    cap = TimedCapture(times, fps)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    with utils.VideoFrameSampler(Path("unused"), interval) as sampler:
        windows = [[frame.timestamp for frame in window] for window in sampler]
    assert cap.released
    return windows, sampler


@pytest.mark.parametrize("fps", [24000 / 1001, 30000 / 1001, 29.97])
def test_fractional_fps_samples_absolute_second_grid_without_drift(monkeypatch, fps):
    times = [index / fps for index in range(math.ceil(101 * fps))]
    windows, sampler = sample(monkeypatch, times, 1, fps)
    heads = [window[0] for window in windows]
    assert len(heads) == 101
    for second, actual in enumerate(heads):
        assert second - 1e-9 <= actual < second + 1 / fps + 1e-9
    assert heads[-1] == pytest.approx(math.ceil(100 * fps) / fps)
    assert sampler.estimated_timestamps == 0


def test_vfr_uses_actual_times_and_drains_partial_windows(monkeypatch):
    windows, _ = sample(monkeypatch, [0, 0.45, 1.02, 1.48, 2.05, 2.45], 1)
    assert windows == [[0, 1.02, 2.05], [1.02, 2.05], [2.05]]


def test_actual_99_6_is_never_stamped_as_deadline_99(monkeypatch):
    windows, _ = sample(monkeypatch, [0, 99.6, 100.2], 1)
    assert handlers._format_timestamp(windows[1][0]) == "00:01:39.600"


def test_large_pts_gap_never_duplicates_a_frame(monkeypatch):
    windows, _ = sample(monkeypatch, [0, 5.3, 5.4], 1)
    assert windows == [[0, 5.3], [5.3]]


def test_subframe_interval_and_zero_interval_keep_each_frame_once(monkeypatch):
    for interval in [0, 0.001]:
        windows, _ = sample(monkeypatch, [0, 0.04, 0.17], interval)
        assert windows == [[0, 0.04, 0.17], [0.04, 0.17], [0.17]]


@pytest.mark.parametrize("bad_time", [float("nan"), float("inf"), -1, 0.03, 0.04])
def test_invalid_or_backward_pts_is_estimated_then_recovers(monkeypatch, bad_time, caplog):
    windows, sampler = sample(monkeypatch, [0, 0.04, bad_time, 0.13], 0)
    assert [window[0] for window in windows] == [0, 0.04, 0.08, 0.13]
    assert sampler.estimated_timestamps == 1
    assert "estimating" in caplog.text


def test_face_intervals_follow_actual_sample_adjacency():
    def frame(ts):
        return ([{"category": "person", "name": "A"}], None, None, None, ts)

    # Jitter in real timestamps must not split consecutive face observations.
    result = handlers._merge_person_timelines(
        [frame(0), frame(1.02), frame(2.05)], 1, [0, 1.02, 2.05]
    )
    assert [row["timestamp"] for row in result] == ["00:00:00.000~00:00:02.050"]
    # An actual sampled frame with a missing result must break the interval.
    result = handlers._merge_person_timelines([frame(0), frame(2.05)], 1, [0, 1.02, 2.05])
    assert len(result) == 2


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="requires FFmpeg for real PTS reference",
)
@pytest.mark.parametrize("start_offset", [0, 5])
def test_real_vfr_decoder_matches_ffprobe_presentation_timestamps(tmp_path, start_offset):
    path = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=25",
            "-vf",
            r"setpts=if(lt(N\,4)\,N*0.04/TB\,(0.12+(N-3)*0.6)/TB)",
            "-frames:v",
            "10",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-bf",
            "2",
            "-output_ts_offset",
            str(start_offset),
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    reference = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
    )
    times = [
        float(frame["best_effort_timestamp_time"]) - start_offset for frame in reference["frames"]
    ]
    assert len({round(b - a, 3) for a, b in zip(times, times[1:])}) > 1
    with utils.VideoFrameSampler(path, 0) as sampler:
        actual = [window[0].timestamp for window in sampler]
    assert actual == pytest.approx(times, abs=1e-6)
    assert sampler.estimated_timestamps == 0
    expected = []
    deadline = 0
    for timestamp in times:
        if timestamp >= deadline:
            expected.append(timestamp)
            deadline = math.floor(timestamp) + 1
    with utils.VideoFrameSampler(path, 1) as sampler:
        actual = [window[0].timestamp for window in sampler]
    assert actual == pytest.approx(expected, abs=1e-6)
