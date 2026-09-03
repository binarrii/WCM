import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from scripts import mark_video_timeline as timeline


def tilde_path(path):
    # Exercise real expanduser without changing HOME or writing to the home directory.
    return f"~/{os.path.relpath(path, Path.home())}"


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js is required for template UI tests")
def test_template_navigation_behavior():
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("video_timeline_ui.test.cjs"))],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "value,expected",
    [
        ("00:00:00.000", 0),
        ("01:02:03.4", 3723400),
        ("00:03:19.000", 199000),
        ("0.9995", 1000),
        (6, 6000),
    ],
)
def test_timestamp_parsing(value, expected):
    assert timeline.timestamp_ms(value) == expected


@pytest.mark.parametrize("value", [None, True, -1, "nan", "inf", "00:60:01", "00:00:60", "bad"])
def test_invalid_timestamp_is_not_silently_skipped(value):
    with pytest.raises(ValueError):
        timeline.timestamp_ms(value)


def test_grouping_sorting_and_deduplication():
    first = {"timestamp": "00:00:02.500", "category": "A", "description": "one"}
    markers = timeline.normalize_results(
        [
            first,
            {"timestamp": 1, "category": "B", "description": "two"},
            first.copy(),
            {**first, "description": "other"},
        ],
        10000,
    )
    assert [m["time_ms"] for m in markers] == [1000, 2500]
    assert len(markers[1]["findings"]) == 2
    chapters = timeline.make_chapters(markers, 10000)
    assert [(c["start_ms"], c["end_ms"]) for c in chapters] == [
        (0, 1000),
        (1000, 2500),
        (2500, 10000),
    ]


def test_ranges_keep_distinct_endpoints_and_other_points_unmerged():
    items = [
        {"timestamp": "00:00:06.000~00:00:07.000", "category": "person", "description": "甲"},
        {"timestamp": "00:00:06.000~00:00:08.000", "category": "person", "description": "乙"},
        {"timestamp": 6, "category": "visual", "description": "review"},
        {"timestamp": 7, "category": "visual", "description": "review"},
    ]
    markers = timeline.normalize_results(items, 10000)
    assert [(m["time_ms"], m["end_time_ms"]) for m in markers] == [
        (6000, 6000),
        (6000, 7000),
        (6000, 8000),
        (7000, 7000),
    ]
    assert markers[1]["timestamp"] == "00:00:06.000~00:00:07.000"
    chapters = timeline.make_chapters(markers, 10000)
    assert [(c["start_ms"], c["end_ms"]) for c in chapters] == [
        (0, 6000),
        (6000, 7000),
        (7000, 10000),
    ]
    assert "~00:00:07.000" in chapters[1]["title"]
    assert "~00:00:08.000" in chapters[1]["title"]
    assert "甲" in chapters[1]["title"] and "乙" in chapters[1]["title"]


@pytest.mark.parametrize("value", ["2~1", "~1", "1~", "1~2~3", "-1~2", "1~nan", "1~11"])
def test_invalid_ranges_fail_instead_of_silently_changing_markers(value):
    with pytest.raises(ValueError):
        timeline.normalize_results([{"timestamp": value}], 10000)


def test_range_can_end_at_duration_and_zero_length_is_a_point():
    markers = timeline.normalize_results([{"timestamp": "0~0"}, {"timestamp": "1~10"}], 10000)
    assert markers[0]["timestamp"] == "00:00:00.000"
    assert markers[1]["end_time_ms"] == 10000


def test_same_interval_for_different_people_stays_separate():
    first = {"timestamp": "6~7", "category": "person", "description": "甲"}
    markers = timeline.normalize_results(
        [first, first.copy(), {**first, "description": "乙"}], 10000
    )
    assert len(markers) == 2
    assert {m["findings"][0]["description"] for m in markers} == {"甲", "乙"}
    assert len(timeline.make_chapters(markers, 10000)) == 2  # Intro + one chapter start.


@pytest.mark.parametrize(
    "payload", [{"detail": "error"}, {"status": "error", "results": []}, [{}], ["bad"]]
)
def test_invalid_api_payload_is_not_reported_as_clean(payload):
    with pytest.raises(ValueError):
        timeline.normalize_results(payload, 10000)


def test_out_of_range_fails_and_zero_marker_is_retained():
    with pytest.raises(ValueError, match="超出"):
        timeline.normalize_results([{"timestamp": 10}], 10000)
    markers = timeline.normalize_results({"results": [{"timestamp": 0}]}, 1000)
    chapters = timeline.make_chapters(markers, 1000)
    assert len(chapters) == 1
    assert chapters[0]["start_ms"] == 0
    assert timeline.normalize_results([], 1000) == []
    assert "无审核标记" in timeline.make_chapters([], 1000)[0]["title"]


def test_html_and_metadata_escape_untrusted_results():
    malicious = '</script><script>alert("bad")</script> & # ; = \\ \nnext'
    markers = timeline.normalize_results(
        [{"timestamp": 1, "category": "other", "description": malicious}], 3000
    )
    rendered = timeline.render_review("<img src=x onerror=alert(1)>", markers, 3000)
    assert malicious not in rendered
    assert "<img src=x" not in rendered
    assert "\\u003c/script\\u003e" in rendered
    assert "text.textContent=details" in rendered
    metadata = timeline.metadata_text(timeline.make_chapters(markers, 3000))
    assert "\\#" in metadata and "\\;" in metadata and "\\=" in metadata
    assert "\nnext" not in metadata


@pytest.mark.parametrize("use_tilde", [False, True])
def test_existing_output_directory_is_never_overwritten(tmp_path, monkeypatch, use_tilde):
    monkeypatch.setattr(timeline.shutil, "which", lambda name: name)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "marked.mp4"
    sentinel.write_bytes(b"original")
    original_mkdir = Path.mkdir

    def checked_mkdir(path, *args, **kwargs):
        assert path == existing.resolve(), "Output path must expand before any directory creation"
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", checked_mkdir)
    args = timeline.build_parser().parse_args(
        [
            "source.mp4",
            "--results",
            "existing.json",
            "--output-dir",
            tilde_path(existing) if use_tilde else str(existing),
        ]
    )
    with pytest.raises(FileExistsError):
        timeline.execute(args)
    assert sentinel.read_bytes() == b"original"


@pytest.mark.parametrize(
    "flag,value",
    [("--sample-interval", "0"), ("--threshold", "nan"), ("--top-k", "11"), ("--timeout", "-1")],
)
def test_invalid_cli_options_fail_before_network_or_outputs(tmp_path, flag, value):
    output = tmp_path / "out"
    args = timeline.build_parser().parse_args(
        ["https://example.com/video.mp4", "--output-dir", str(output), flag, value]
    )
    with pytest.raises(ValueError):
        timeline.execute(args)
    assert not output.exists()


def test_local_source_requires_existing_analysis_json():
    args = timeline.build_parser().parse_args(["private.mp4"])
    with pytest.raises(ValueError, match="不会自动上传"):
        timeline.execute(args)


@pytest.fixture
def sample_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not installed")
    video = tmp_path / "source.mp4"
    timeline.run_media_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=128x96:r=10:d=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ]
    )
    return video


def test_complete_export_and_stream_copy(sample_video, tmp_path):
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            [
                {
                    "timestamp": "00:00:01.000~00:00:02.000",
                    "category": "复核",
                    "description": "引号 ; # = \\ \n换行",
                },
                {"timestamp": "00:00:02.000", "category": "复核", "description": "另一个标记"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    before = sample_video.read_bytes()
    args = timeline.build_parser().parse_args(
        [
            str(sample_video),
            "--results",
            str(results),
            "--output-dir",
            str(tmp_path / "review"),
        ]
    )
    output = timeline.execute(args)
    assert sample_video.read_bytes() == before
    assert {p.name for p in output.iterdir()} == {
        "marked.mp4",
        "review.html",
        "analysis.json",
        "markers.json",
        "chapters.ffmetadata",
    }
    probe = timeline.probe_video(output / "marked.mp4", "ffprobe")
    assert [float(c["start_time"]) for c in probe["chapters"]] == [0, 1, 2]
    assert {s["codec_type"] for s in probe["streams"]} >= {"audio", "video"}

    def media_hash(path):
        return timeline.run_media_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v",
                "-map",
                "0:a",
                "-c",
                "copy",
                "-f",
                "streamhash",
                "-",
            ]
        )

    assert media_hash(sample_video) == media_hash(output / "marked.mp4")


def test_cli_expands_tilde_for_all_disk_paths_without_shell(sample_video, tmp_path):
    inputs = tmp_path / "clips ~ archive"
    inputs.mkdir()
    source = inputs / "测试 video.mp4"
    shutil.copyfile(sample_video, source)
    results = inputs / "analysis results.json"
    results.write_text('[{"timestamp": 1, "category": "test"}]', encoding="utf-8")
    output = tmp_path / "review output"
    # Keep even an incorrectly resolved literal ~ path inside this test's temporary tree.
    working_directory = tmp_path.joinpath(*(["work"] * len(Path.home().parts)))
    working_directory.mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(timeline.__file__).resolve()),
            tilde_path(source),
            "--results",
            tilde_path(results),
            "--output-dir",
            tilde_path(output),
        ],
        cwd=working_directory,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert str(output.resolve()) in result.stdout
    assert (output / "review.html").is_file()
    assert len(timeline.probe_video(output / "marked.mp4", "ffprobe")["chapters"]) == 2
    assert source.read_bytes() == sample_video.read_bytes()
    assert not (working_directory / "~").exists()


def test_live_api_mode_serializes_parameters_without_retries(sample_video, tmp_path, monkeypatch):
    payloads = []

    def respond(request):
        assert request.url.path == "/api/v1/analyze_media"
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(timeline.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        timeline,
        "download_video",
        lambda url, destination, max_bytes: shutil.copyfile(sample_video, destination),
    )
    args = timeline.build_parser().parse_args(
        [
            "https://example.com/~media/video.mp4",
            "--api-url",
            "http://test/api/v1/analyze_media",
            "--threshold",
            "0.9",
            "--sample-interval",
            "2",
            "--output-dir",
            str(tmp_path / "api-review"),
        ]
    )
    output = timeline.execute(args)
    assert payloads == [{"url": args.video, "threshold": 0.9, "sample_interval": 2.0, "top_k": 10}]
    assert json.loads((output / "markers.json").read_text())["markers"] == []
