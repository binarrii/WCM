import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import httpx
import pytest

from api import media_source as media
from api import utils
from wcm_facerec.config import settings


@pytest.fixture
def media_http(monkeypatch):
    routes, requests = {}, []
    original = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        value = routes.get(str(request.url))
        if callable(value):
            return value(request)
        if value is None:
            return httpx.Response(404)
        return httpx.Response(200, content=value)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw)
    )
    monkeypatch.setattr(settings, "cluster_enabled", False)
    monkeypatch.setattr(settings, "video_min_free_disk_mb", 1)
    return routes, requests


@pytest.fixture(scope="module")
def video_files(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    root = tmp_path_factory.mktemp("media-fixtures")

    def run(*args):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", *map(str, args)], check=True)

    run(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x90:rate=25:duration=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3",
        "-c:v",
        "libx264",
        "-g",
        "25",
        "-threads",
        "1",
        "-c:a",
        "libmp3lame",
        "-f",
        "mpegts",
        root / "source.ts",
    )
    for segment_type in ("mpegts", "fmp4"):
        directory = root / segment_type
        directory.mkdir()
        run(
            "-i",
            root / "source.ts",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-f",
            "hls",
            "-hls_time",
            "1",
            "-hls_playlist_type",
            "vod",
            "-hls_segment_type",
            segment_type,
            directory / "index.m3u8",
        )
    return root


async def prepare(url, path, max_size=10000000, **kwargs):
    return await media.prepare_video(
        url, path, max_size, downloader=utils._download_video_file_async, **kwargs
    )


@pytest.mark.asyncio
async def test_signed_ts_and_unknown_content_are_videos(media_http):
    routes, _ = media_http
    assert await media.is_video_url("https://source/video.TS?token=abc")
    assert await media.is_video_url("https://source/stream.m3u8?token=abc")
    assert not await media.is_video_url("https://source/img.jpg?token=abc")
    routes["https://source/opaque"] = b"#EXTM3U\n"
    assert await media.is_video_url("https://source/opaque")
    routes["https://source/image"] = b"\xff\xd8\xff" + b"0" * 5000
    assert not await media.is_video_url("https://source/image")


@pytest.mark.asyncio
async def test_ts_remux_preserves_frames_and_maps_player_clock(media_http, video_files, tmp_path):
    routes, _ = media_http
    routes["https://source/video.ts?token=x"] = (video_files / "source.ts").read_bytes()
    output = tmp_path / "review.mp4"
    metadata = await prepare("https://source/video.ts?token=x", output)
    assert metadata["video_codec"] == "h264" and not metadata["video_transcoded"]
    info, video, duration = await media.probe(output)
    assert any(s["codec_name"] == "aac" for s in info["streams"])
    source = cv2.VideoCapture(str(video_files / "source.ts"))
    result = cv2.VideoCapture(str(output))
    count = 0
    try:
        while True:
            a, before = source.read()
            b, after = result.read()
            assert a == b
            if not a:
                break
            assert (before == after).all()
            assert abs(result.get(cv2.CAP_PROP_POS_MSEC) / 1000 - count / 25) < 0.002
            count += 1
    finally:
        source.release()
        result.release()
    assert count == 75
    frames = json.loads(
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
                str(output),
            ]
        )
    )["frames"]
    for i, frame in enumerate(frames):
        assert (
            abs(
                float(frame["best_effort_timestamp_time"])
                - metadata["video_start_seconds"]
                - i / 25
            )
            < 0.002
        )
    assert not list(tmp_path.glob("wcm-ingest-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("segment_type", ["mpegts", "fmp4"])
async def test_hls_vod_localization_master_relative_urls(
    media_http, video_files, tmp_path, segment_type
):
    routes, requests = media_http
    root = video_files / segment_type
    routes["https://source/master?token=x"] = (
        b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100000,RESOLUTION=160x90\nmedia/index.m3u8\n"
    )
    for file in root.iterdir():
        routes[f"https://source/media/{file.name}"] = file.read_bytes()
    metadata = await prepare("https://source/master?token=x", tmp_path / "out.mp4")
    assert metadata["source_kind"] == "hls"
    assert metadata["selection"]["resolution"] == (160, 90)
    assert 2.9 <= metadata["duration_seconds"] <= 3.3
    assert metadata["input_bytes"] > len(routes["https://source/master?token=x"])
    assert not any("token=x" in str(r.url) for r in requests[1:])


@pytest.mark.asyncio
async def test_live_rejected_before_fetching_segments(media_http, tmp_path):
    routes, requests = media_http
    routes["https://source/live.m3u8"] = b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n1.ts\n"
    with pytest.raises(media.MediaSourceError, match="直播"):
        await prepare("https://source/live.m3u8", tmp_path / "out.mp4")
    assert len(requests) == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_hls_cumulative_limit_and_cleanup(media_http, tmp_path):
    routes, requests = media_http
    routes["https://source/v.m3u8"] = (
        b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\na.ts\n#EXTINF:1,\nb.ts\n#EXT-X-ENDLIST\n"
    )
    routes["https://source/a.ts"] = b"a" * 1000
    routes["https://source/b.ts"] = b"b" * 1000
    with pytest.raises(media.MediaSourceError, match="大小限制"):
        await prepare("https://source/v.m3u8", tmp_path / "out.mp4", 1500)
    assert len(requests) == 3
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_hls_rejects_local_uris_and_missing_byte_ranges(media_http, tmp_path):
    routes, _ = media_http
    routes["https://source/v.m3u8"] = (
        b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n"
    )
    with pytest.raises(media.MediaSourceError, match="HTTP"):
        await prepare("https://source/v.m3u8", tmp_path / "out.mp4")
    with pytest.raises(media.MediaSourceError, match="连续"):
        media.HlsDownload.byte_range("100", None)


@pytest.mark.asyncio
async def test_hls_byterange_validates_server_response(media_http, tmp_path):
    routes, _ = media_http
    routes["https://source/s"] = b"0123456789"
    async with httpx.AsyncClient() as client:
        downloader = media.HlsDownload(client, tmp_path, 1000, 0, None)
        with pytest.raises(media.MediaSourceError, match="字节范围"):
            await downloader.fetch("https://source/s", byte_range=(2, 4))
        routes["https://source/s"] = lambda req: httpx.Response(
            206, content=b"2345", headers={"Content-Range": "bytes 2-5/10"}
        )
        path, _ = await downloader.fetch("https://source/s", byte_range=(2, 4))
        assert path.read_bytes() == b"2345"


@pytest.mark.asyncio
async def test_process_cancel_kills_child_and_drains(monkeypatch, tmp_path):
    started = asyncio.Event()
    original = asyncio.create_subprocess_exec
    child = None

    async def start(*args, **kwargs):
        nonlocal child
        child = await original(*args, **kwargs)
        started.set()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", start)
    import sys

    task = asyncio.create_task(
        media.media_command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=40)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    assert child.returncode is not None


@pytest.mark.asyncio
async def test_duration_and_disk_limits(media_http, video_files, tmp_path, monkeypatch):
    routes, _ = media_http
    routes["https://source/v.ts"] = (video_files / "source.ts").read_bytes()
    monkeypatch.setattr(settings, "max_video_duration_seconds", 1)
    with pytest.raises(media.MediaSourceError, match="时长"):
        await prepare("https://source/v.ts", tmp_path / "out.mp4")
    monkeypatch.setattr(settings, "video_min_free_disk_mb", 10**12)
    with pytest.raises(media.MediaSourceError, match="磁盘"):
        await prepare("https://source/v.ts", tmp_path / "out.mp4")


@pytest.mark.asyncio
async def test_aes128_segments_are_localized_and_decrypted(media_http, video_files, tmp_path):
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    routes, requests = media_http
    root = video_files / "mpegts"
    key = b"test-hls-key-128"
    routes["https://source/key"] = key
    playlist = (
        (root / "index.m3u8")
        .read_text()
        .replace("#EXTM3U\n", '#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI="key"\n')
    )
    routes["https://source/index.m3u8"] = playlist.encode()
    for index, file in enumerate(sorted(root.glob("*.ts"))):
        padder = padding.PKCS7(128).padder()
        content = padder.update(file.read_bytes()) + padder.finalize()
        encryptor = Cipher(algorithms.AES(key), modes.CBC(index.to_bytes(16, "big"))).encryptor()
        routes[f"https://source/{file.name}"] = encryptor.update(content) + encryptor.finalize()
    result = await prepare("https://source/index.m3u8", tmp_path / "out.mp4")
    assert 2.9 <= result["duration_seconds"] <= 3.3
    assert sum(str(r.url) == "https://source/key" for r in requests) == 1


@pytest.mark.asyncio
async def test_alternate_audio_and_byte_range_hls(media_http, video_files, tmp_path):
    routes, _ = media_http
    for name, args in (
        ("video", ["-an", "-c:v", "copy"]),
        ("audio", ["-vn", "-c:a", "aac"]),
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-i",
                str(video_files / "source.ts"),
                *args,
                "-f",
                "hls",
                "-hls_time",
                "1",
                "-hls_flags",
                "single_file",
                "-hls_playlist_type",
                "vod",
                str(tmp_path / f"{name}.m3u8"),
            ],
            check=True,
        )
    routes["https://source/master.m3u8"] = (
        b'#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a",NAME="Main",DEFAULT=YES,URI="audio.m3u8"\n#EXT-X-STREAM-INF:BANDWIDTH=100000,RESOLUTION=160x90,AUDIO="a"\nvideo.m3u8\n'
    )
    for file in tmp_path.glob("*.m3u8"):
        routes[f"https://source/{file.name}"] = file.read_bytes()

    def ranged(data):
        def respond(request):
            start, end = map(int, request.headers["Range"][6:].split("-"))
            return httpx.Response(
                206,
                content=data[start : end + 1],
                headers={"Content-Range": f"bytes {start}-{end}/{len(data)}"},
            )

        return respond

    for file in tmp_path.glob("*.ts"):
        routes[f"https://source/{file.name}"] = ranged(file.read_bytes())
    result = await prepare("https://source/master.m3u8", tmp_path / "out.mp4")
    info, _, _ = await media.probe(tmp_path / "out.mp4")
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert result["source_kind"] == "hls"


@pytest.mark.asyncio
async def test_non_browser_video_is_transcoded(media_http, video_files, tmp_path):
    routes, _ = media_http
    source = tmp_path / "mpeg2.ts"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(video_files / "source.ts"),
            "-c:v",
            "mpeg2video",
            "-an",
            str(source),
        ],
        check=True,
    )
    routes["https://source/v.ts"] = source.read_bytes()
    result = await prepare("https://source/v.ts", tmp_path / "out.mp4")
    assert result["video_transcoded"] and result["video_codec"] == "h264"
