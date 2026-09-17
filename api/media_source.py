"""Bounded HTTP media ingestion. FFmpeg only sees localized, finite inputs."""

import asyncio
import base64
import contextlib
import json
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
import m3u8
import numpy as np

from wcm_facerec.cluster import cluster_slot, drain_task
from wcm_facerec.config import settings

from .media_decode_check import SAMPLE_HEIGHT, SAMPLE_WIDTH

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".ts",
    ".mts",
    ".m2ts",
    ".mpeg",
    ".mpg",
    ".m3u8",
    ".m3u",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
MANIFEST_LIMIT = 2 * 1024 * 1024


class MediaSourceError(ValueError):
    """Permanent source/limit errors must not repeat the whole review."""


def http_url(url):
    parts = urlsplit(str(url))
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise MediaSourceError("媒体地址仅支持无内嵌账号密码的 HTTP(S) URL")
    return str(url)


def video_limit_bytes():
    return settings.max_video_size_mb * 1024 * 1024


def size_error(actual, maximum):
    return MediaSourceError(
        f"视频超过大小限制：已知或已读取 {actual / 1048576:.2f} MiB，上限 {maximum / 1048576:.2f} MiB"
    )


def check_disk(path):
    if shutil.disk_usage(Path(path).parent).free < settings.video_min_free_disk_mb * 1048576:
        raise MediaSourceError("视频临时磁盘剩余空间不足，请释放空间后重试")


async def is_video_url(url):
    """Extensions are hints; unknown URLs are sniffed with a bounded GET."""
    suffix = Path(urlsplit(url).path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return True
    if suffix in IMAGE_EXTENSIONS:
        return False
    http_url(url)
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        async with client.stream(
            "GET", url, headers={"Range": "bytes=0-65535", "Accept-Encoding": "identity"}
        ) as response:
            response.raise_for_status()
            http_url(response.url)
            content_type = response.headers.get("content-type", "").lower()
            if content_type.startswith("image/"):
                return False
            if content_type.startswith("video/") or "mpegurl" in content_type:
                return True
            data = bytearray()
            async for chunk in response.aiter_bytes(4096):
                data.extend(chunk)
                if len(data) >= 4096:
                    break
            # Unknown binary sources are probed as video, never decoded as images.
            return not (
                data.startswith((b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM"))
                or data[8:12] == b"WEBP"
            )


async def media_command(args, *, output=None, timeout=60):
    """Bounded diagnostics, disk/output watchdog, and drained process cancellation."""
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = bytearray(), bytearray()

    async def read(stream, target, limit):
        while chunk := await stream.read(65536):
            target.extend(chunk)
            if len(target) > limit:
                del target[: len(target) - limit]

    readers = [
        asyncio.create_task(read(process.stdout, stdout, 2 * 1024 * 1024)),
        asyncio.create_task(read(process.stderr, stderr, 16384)),
    ]
    try:
        deadline = time.monotonic() + timeout
        while process.returncode is None:
            if time.monotonic() >= deadline:
                raise MediaSourceError("媒体准备超时，请检查视频时长或提高媒体准备时限")
            if output:
                check_disk(output)
                if (
                    output.exists()
                    and output.stat().st_size > settings.max_video_output_mb * 1048576
                ):
                    raise size_error(output.stat().st_size, settings.max_video_output_mb * 1048576)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), 0.2)
        await asyncio.gather(*readers)
        if process.returncode:
            # FFmpeg diagnostics may contain signed URLs or local paths.
            raise MediaSourceError("视频探测或转换失败：媒体损坏、编码不支持或时间戳异常")
        return bytes(stdout)
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        cleanup = asyncio.create_task(process.wait())
        await drain_task(cleanup)
        for task in readers:
            await drain_task(task)


def input_options(path):
    options = ["-protocol_whitelist", "file,crypto"]
    if Path(path).suffix == ".m3u8":
        # This playlist is generated below; all URIs point at downloaded files.
        options += ["-allowed_extensions", "ALL"]
    return options


async def probe(path):
    data = await media_command(
        [
            "ffprobe",
            "-v",
            "error",
            *input_options(path),
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        result = json.loads(data)
        video = next(
            s
            for s in result["streams"]
            if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")
        )
        duration = float(result.get("format", {}).get("duration") or video.get("duration") or 0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError()
    except (ValueError, KeyError, StopIteration):
        raise MediaSourceError("媒体没有可用视频轨道或无法确定完整时长") from None
    if duration > settings.max_video_duration_seconds:
        raise MediaSourceError(
            f"视频时长 {duration:.1f} 秒超过上限 {settings.max_video_duration_seconds} 秒"
        )
    return result, video, duration


async def verify_decoder(path, info, video, duration):
    """Cross-check real decoder pixels, accepting genuine black/static footage.

    A successful VideoCapture.read() can still return all-zero pixels when its
    bundled swscale cannot convert an input. Independent FFmpeg references catch
    that failure before scene deduplication can turn it into a successful review.
    Both decoder and reference processes are bounded and killed on cancellation.
    """
    video_duration = float(video.get("duration") or duration)
    samples = json.loads(
        await media_command(
            [sys.executable, "-m", "api.media_decode_check", str(path), str(video_duration)],
            timeout=60,
        )
    )
    video_start = float(video.get("start_time") or 0)
    format_start = float(info.get("format", {}).get("start_time") or 0)
    for sample in samples["samples"]:
        # OpenCV's clock starts at the first video frame; FFmpeg input seeking
        # starts at format.start_time. Stay just before the exact frame PTS so
        # floating point rounding cannot select the following frame at a cut.
        target = max(0, sample["pts"] + video_start - format_start - 0.0005)
        reference = await media_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-threads",
                "2",
                "-ss",
                str(target),
                *input_options(path),
                "-i",
                str(path),
                "-map",
                f"0:{video['index']}",
                "-frames:v",
                "1",
                "-vf",
                f"scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT}:flags=area",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            timeout=30,
        )
        actual = base64.b64decode(sample["pixels"], validate=True)
        expected_bytes = SAMPLE_WIDTH * SAMPLE_HEIGHT * 3
        if len(actual) != expected_bytes or len(reference) != expected_bytes:
            raise MediaSourceError("视频抽帧校验失败：未获得完整画面，已停止审核")
        actual_pixels = np.frombuffer(actual, np.uint8).astype(np.int16)
        reference_pixels = np.frombuffer(reference, np.uint8).astype(np.int16)
        difference = float(np.abs(actual_pixels - reference_pixels).mean())
        if difference > 32 or (not actual_pixels.any() and reference_pixels.mean() > 2):
            raise MediaSourceError(
                f"视频抽帧校验失败：第 {sample['pts']:.2f} 秒的解码画面异常，已停止审核"
            )
    return {"decoder_verified": True, "decoder_version": samples["decoder_version"]}


class HlsDownload:
    def __init__(self, client, directory, max_size, downloaded, on_progress):
        self.client, self.directory, self.maximum = client, directory, max_size
        self.downloaded, self.on_progress = downloaded, on_progress
        self.counter = 0
        self.keys = {}

    async def fetch(self, url, *, limit=None, byte_range=None, suffix=".bin"):
        url = http_url(url)
        headers = {"Accept-Encoding": "identity"}
        if byte_range:
            start, length = byte_range
            headers["Range"] = f"bytes={start}-{start + length - 1}"
        path = self.directory / f"part-{self.counter}{suffix}"
        self.counter += 1
        size = 0
        async with self.client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            http_url(response.url)
            if byte_range:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("content-range", "")
                )
                if (
                    response.status_code != 206
                    or not match
                    or (int(match[1]), int(match[2])) != (start, start + length - 1)
                ):
                    raise MediaSourceError("HLS 分片服务器未正确响应字节范围请求")
            with path.open("wb") as target:
                async for chunk in response.aiter_bytes(65536):
                    size += len(chunk)
                    self.downloaded += len(chunk)
                    if self.downloaded > self.maximum:
                        raise size_error(self.downloaded, self.maximum)
                    if (limit is not None and size > limit) or (byte_range and size > length):
                        raise MediaSourceError("HLS 播放列表、密钥或分片长度异常")
                    check_disk(path)
                    target.write(chunk)
                    if self.on_progress:
                        self.on_progress(self.downloaded, None)
            if byte_range and size != length:
                raise MediaSourceError("HLS 分片不完整")
            return path, str(response.url)

    def parse(self, text, url):
        if not text.lstrip("\ufeff \r\n").startswith("#EXTM3U"):
            raise MediaSourceError("HLS 播放列表内容无效")
        try:
            return m3u8.loads(text, uri=url)
        except (ValueError, TypeError, KeyError):
            raise MediaSourceError("HLS 播放列表解析失败") from None

    async def playlist(self, url):
        path, final_url = await self.fetch(url, limit=MANIFEST_LIMIT, suffix=".txt")
        return self.parse(path.read_text(encoding="utf-8-sig"), final_url), final_url

    async def localize(self, playlist, url, name):
        if not playlist.is_endlist:
            raise MediaSourceError(
                "检测到仍在更新的 HLS 直播流；本次支持完整点播，请提供已结束的播放列表或录制文件"
            )
        segments = playlist.segments
        if not segments or len(segments) > settings.hls_max_segments:
            raise MediaSourceError("HLS 分片数量为空或超过配置上限")
        durations = [float(s.duration) for s in segments]
        if any(not math.isfinite(d) or d <= 0 for d in durations):
            raise MediaSourceError("HLS 分片时长无效")
        if sum(durations) > settings.max_video_duration_seconds:
            raise MediaSourceError("HLS 总时长超过配置上限")
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:7",
            f"#EXT-X-TARGETDURATION:{math.ceil(max(durations))}",
            f"#EXT-X-MEDIA-SEQUENCE:{playlist.media_sequence or 0}",
        ]
        previous_uri, previous_end = None, None
        previous_map = None
        for segment in segments:
            if segment.gap_tag:
                raise MediaSourceError("HLS 包含缺失分片，无法保证完整审核")
            if segment.discontinuity:
                lines.append("#EXT-X-DISCONTINUITY")
            key = segment.key
            if key and key.method != "NONE":
                if key.method != "AES-128" or key.keyformat not in (None, "identity"):
                    raise MediaSourceError(
                        "暂不支持 DRM 或 SAMPLE-AES 加密 HLS，请提供可直接解码的视频"
                    )
                key_url = http_url(urljoin(url, key.uri))
                if key_url not in self.keys:
                    key_path, _ = await self.fetch(key_url, limit=16, suffix=".key")
                    if key_path.stat().st_size != 16:
                        raise MediaSourceError("HLS AES-128 密钥长度无效")
                    self.keys[key_url] = key_path.name
                iv = key.iv
                if iv and not re.fullmatch(r"0[xX][0-9a-fA-F]{1,32}", iv):
                    raise MediaSourceError("HLS 加密 IV 无效")
                lines.append(
                    f'#EXT-X-KEY:METHOD=AES-128,URI="{self.keys[key_url]}"'
                    + (f",IV={iv}" if iv else "")
                )
            else:
                lines.append("#EXT-X-KEY:METHOD=NONE")
            init = segment.init_section
            if init:
                identity = (
                    init.uri,
                    init.byterange,
                    key.uri if key else None,
                    key.iv if key else None,
                )
                if identity != previous_map:
                    init_range = self.byte_range(init.byterange, None)
                    init_path, _ = await self.fetch(
                        urljoin(url, init.uri), byte_range=init_range, suffix=".mp4"
                    )
                    lines.append(f'#EXT-X-MAP:URI="{init_path.name}"')
                    previous_map = identity
            segment_url = http_url(urljoin(url, segment.uri))
            byte_range = self.byte_range(
                segment.byterange, previous_end if segment_url == previous_uri else None
            )
            path, _ = await self.fetch(
                segment_url, byte_range=byte_range, suffix=".m4s" if init else ".ts"
            )
            previous_uri = segment_url
            previous_end = sum(byte_range) if byte_range else None
            lines += [f"#EXTINF:{segment.duration},", path.name]
        lines.append("#EXT-X-ENDLIST")
        path = self.directory / f"{name}.m3u8"
        path.write_text("\n".join(lines) + "\n")
        return path

    @staticmethod
    def byte_range(value, previous_end):
        if not value:
            return None
        match = re.fullmatch(r"(\d+)(?:@(\d+))?", value)
        if not match:
            raise MediaSourceError("HLS 字节范围无效")
        start = int(match[2]) if match[2] is not None else previous_end
        length = int(match[1])
        if start is None or length <= 0:
            raise MediaSourceError("HLS 隐式字节范围缺少连续的前一分片")
        return start, length

    async def ingest(self, text, url):
        playlist = self.parse(text, url)
        audio_url, selected = None, {}
        for _ in range(5):
            if not playlist.is_variant:
                break
            candidates = [
                p
                for p in playlist.playlists
                if not p.stream_info.resolution
                or p.stream_info.resolution[1] <= settings.hls_max_height
            ]
            if not candidates:
                candidates = list(playlist.playlists)
            if not candidates:
                raise MediaSourceError("HLS 主列表没有可用清晰度")
            variant = max(
                candidates,
                key=lambda p: (
                    (p.stream_info.resolution or (0, 0))[1],
                    p.stream_info.bandwidth or 0,
                ),
            )
            selected = {
                "resolution": variant.stream_info.resolution,
                "bandwidth": variant.stream_info.bandwidth,
            }
            audio = [
                m
                for m in playlist.media
                if m.type == "AUDIO" and m.group_id == variant.stream_info.audio and m.uri
            ]
            if audio:
                rendition = next((m for m in audio if m.default == "YES"), audio[0])
                audio_url = urljoin(url, rendition.uri)
            playlist, url = await self.playlist(urljoin(url, variant.uri))
        else:
            raise MediaSourceError("HLS 播放列表嵌套过深或循环引用")
        paths = [await self.localize(playlist, url, "video")]
        if audio_url:
            audio, audio_url = await self.playlist(audio_url)
            paths.append(await self.localize(audio, audio_url, "audio"))
        return paths, selected


async def prepare_video(
    url, destination, max_size, *, downloader, timeout=900, on_progress=None, on_stage=None
):
    """Download, localize HLS, and remux/transcode into the actual review MP4."""
    http_url(url)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_disk(destination)

    async def prepare():
        with tempfile.TemporaryDirectory(prefix="wcm-ingest-", dir=destination.parent) as folder:
            directory = Path(folder)
            raw = directory / "source.bin"
            response = await downloader(url, raw, max_size, timeout, on_progress=on_progress)
            final_url = (response or {}).get("url", url)
            with raw.open("rb") as source:
                prefix = source.read(64)
            selected = {}
            source_kind = "file"
            input_bytes = raw.stat().st_size
            if prefix.lstrip(b"\xef\xbb\xbf \r\n").startswith(b"#EXTM3U"):
                if input_bytes > MANIFEST_LIMIT:
                    raise MediaSourceError("HLS 播放列表过大")
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(60, connect=15), follow_redirects=True
                ) as client:
                    hls = HlsDownload(client, directory, max_size, input_bytes, on_progress)
                    paths, selected = await hls.ingest(
                        raw.read_text(encoding="utf-8-sig"), final_url
                    )
                    input_bytes = hls.downloaded
                source_kind = "hls"
            else:
                paths = [raw]
            if on_stage:
                await on_stage("preparing")
            info, video, duration = await probe(paths[0])
            audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
            field_order = video.get("field_order", "unknown")
            deinterlace = field_order != "progressive"
            copy_video = (
                video.get("codec_name") == "h264"
                and video.get("pix_fmt") == "yuv420p"
                and not deinterlace
            )
            args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
            for path in paths:
                args += [*input_options(path), "-i", str(path)]
            args += [
                "-map",
                f"0:{video['index']}",
                "-map",
                "1:a:0" if len(paths) > 1 else "0:a:0?",
                "-sn",
                "-dn",
            ]
            args += (
                ["-c:v", "copy"]
                if copy_video
                else [
                    "-vf",
                    # One output frame per input frame preserves review timing.
                    "bwdif=mode=send_frame:parity=auto:deint=interlaced,setfield=prog"
                    if deinterlace
                    else "setfield=prog",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-fps_mode",
                    "passthrough",
                    "-threads",
                    "2",
                ]
            )
            args += [
                "-c:a",
                "copy" if len(paths) == 1 and audio and audio.get("codec_name") == "aac" else "aac",
                "-movflags",
                "+faststart",
                "-max_muxing_queue_size",
                "2048",
                str(destination),
            ]
            # The cluster-wide slot bounds expensive video/audio conversion.
            async with cluster_slot("media-prepare", settings.video_prepare_concurrency):
                await media_command(
                    args, output=destination, timeout=settings.video_prepare_timeout_seconds
                )
                output_info, output_video, output_duration = await probe(destination)
                if output_video.get("field_order") != "progressive":
                    raise MediaSourceError("视频未正确转换为逐行画面，已停止审核")
                decoder = await verify_decoder(
                    destination, output_info, output_video, output_duration
                )
            if abs(output_duration - duration) > max(2, duration * 0.02):
                raise MediaSourceError("转换后视频时长异常，已停止审核以避免遗漏内容")
            if destination.stat().st_size > settings.max_video_output_mb * 1048576:
                raise size_error(destination.stat().st_size, settings.max_video_output_mb * 1048576)
            return {
                "source_kind": source_kind,
                "input_bytes": input_bytes,
                "size_bytes": destination.stat().st_size,
                "duration_seconds": output_duration,
                "video_start_seconds": float(output_video.get("start_time") or 0),
                "width": output_video["width"],
                "height": output_video["height"],
                "video_codec": output_video["codec_name"],
                "video_transcoded": not copy_video,
                "source_field_order": field_order,
                "deinterlaced": deinterlace,
                **decoder,
                "selection": selected,
            }

    try:
        return await asyncio.wait_for(prepare(), settings.video_prepare_timeout_seconds)
    except asyncio.TimeoutError:
        destination.unlink(missing_ok=True)
        raise MediaSourceError("媒体准备超时，请检查源地址或提高媒体准备时限") from None
    except httpx.HTTPStatusError as exc:
        destination.unlink(missing_ok=True)
        code = exc.response.status_code
        if 400 <= code < 500 and code not in {408, 429}:
            raise MediaSourceError(f"媒体源读取失败（HTTP {code}），请检查地址及访问权限") from None
        raise RuntimeError(f"媒体源暂时不可用（HTTP {code}）") from None
    except httpx.RequestError:
        destination.unlink(missing_ok=True)
        raise RuntimeError("媒体源连接失败或读取超时，请检查网络及源服务") from None
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
