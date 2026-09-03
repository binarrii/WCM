#!/usr/bin/env python3
"""Turn /api/v1/analyze_media results into MP4 chapters and a review timeline.

Requires httpx (project dependency), ffmpeg and ffprobe on PATH. Source files
are never modified; each run writes to a new output directory. API findings
are review cues, not independently verified violations or violation durations.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

DEFAULT_API_URL = "http://10.252.25.251:8000/api/v1/analyze_media"


def timestamp_ms(value: object) -> int:
    """Parse seconds or HH:MM:SS[.mmm], rejecting invalid/negative timestamps."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"无效时间戳：{value!r}")
    text = str(value).strip()
    if ":" in text:
        match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)(?:\.(\d{1,3}))?", text)
        if not match:
            raise ValueError(f"时间戳应为 HH:MM:SS.mmm：{text!r}")
        hours, minutes, seconds, fraction = match.groups()
        return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(
            (fraction or "0").ljust(3, "0")
        )
    try:
        seconds = Decimal(text)
        if not seconds.is_finite() or seconds < 0:
            raise ValueError(f"无效时间戳：{text!r}")
        return int((seconds * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except InvalidOperation as exc:
        raise ValueError(f"无效时间戳：{text!r}") from exc


def format_timestamp(milliseconds: int) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def timestamp_range_ms(value: object) -> tuple[int, int]:
    """Accept legacy points and start~end intervals emitted by analyze_media."""
    if isinstance(value, str) and "~" in value:
        parts = value.split("~")
        if len(parts) != 2:
            raise ValueError(f"无效时间区间：{value!r}")
        start, end = (timestamp_ms(part) for part in parts)
        if end < start:
            raise ValueError(f"区间结束时间早于开始时间：{value!r}")
        return start, end
    point = timestamp_ms(value)
    return point, point


def normalize_results(payload: object, duration_ms: int) -> list[dict]:
    """Group only identical spans; never infer intervals from legacy point results."""
    if isinstance(payload, dict) and payload.get("status") not in ("error", "failed"):
        payload = payload.get("results")
    if not isinstance(payload, list):
        raise ValueError("审核结果必须是数组，或包含 results 数组的对象；不能把错误响应当成无标记")
    grouped: dict[tuple[int, int, str, str], list[dict]] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or "timestamp" not in item:
            raise ValueError(f"第 {index + 1} 条结果缺少 timestamp")
        start, end = timestamp_range_ms(item["timestamp"])
        if start >= duration_ms or end > duration_ms:
            raise ValueError(f"第 {index + 1} 条时间戳超出视频时长：{item['timestamp']}")
        category = item.get("category") or "未分类"
        description = item.get("description") or ""
        if not isinstance(category, str) or not isinstance(description, str):
            raise ValueError(f"第 {index + 1} 条 category/description 必须是文本")
        finding = {"category": category, "description": description}
        # Different people keep distinct interval entries, even with identical bounds.
        key = (start, end, category if end > start else "", description if end > start else "")
        findings = grouped.setdefault(key, [])
        if finding not in findings:
            findings.append(finding)
    return [
        {
            "timestamp": format_timestamp(start)
            + (f"~{format_timestamp(end)}" if end > start else ""),
            "time_ms": start,
            "end_time_ms": end,
            "findings": findings,
        }
        for (start, end, _, _), findings in sorted(grouped.items())
    ]


def escape_metadata(text: str) -> str:
    # FFmetadata special characters: https://ffmpeg.org/ffmpeg-formats.html#Metadata
    text = text.replace("\r", " ").replace("\n", " ").replace("\x00", "")
    return re.sub(r"([\\=;#])", r"\\\1", text)


def chapter_title(marker: dict) -> str:
    detail = " / ".join(f"{item['category']}: {item['description']}" for item in marker["findings"])
    return " ".join(f"{marker['timestamp']} 待复核 · {detail}".replace("\x00", "").split())[:240]


def make_chapters(markers: list[dict], duration_ms: int) -> list[dict]:
    # A point and several intervals may share a start. MP4 needs one chapter
    # per start, but the HTML/JSON retain their separate spans and full details.
    titles_by_start: dict[int, list[str]] = {}
    for marker in markers:
        titles_by_start.setdefault(marker["time_ms"], []).append(chapter_title(marker))
    chapters = [
        {"start_ms": start, "title": " | ".join(titles)[:240]}
        for start, titles in sorted(titles_by_start.items())
    ]
    if not chapters or chapters[0]["start_ms"] > 0:
        chapters.insert(0, {"start_ms": 0, "title": "开始（此处无审核标记）"})
    for index, chapter in enumerate(chapters):
        chapter["end_ms"] = (
            chapters[index + 1]["start_ms"] if index + 1 < len(chapters) else duration_ms
        )
    return chapters


def metadata_text(chapters: list[dict]) -> str:
    lines = [";FFMETADATA1"]
    for chapter in chapters:
        lines.extend(
            [
                "",
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={chapter['start_ms']}",
                f"END={chapter['end_ms']}",
                f"title={escape_metadata(chapter['title'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def json_for_html(value: object) -> str:
    # API text is untrusted: never let it terminate the JSON script element.
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_review(title: str, markers: list[dict], duration_ms: int) -> str:
    template = Path(__file__).with_name("video_timeline.html").read_text(encoding="utf-8")
    return template.replace("__TITLE__", html.escape(title)).replace(
        "__REVIEW_DATA__", json_for_html({"markers": markers, "duration_ms": duration_ms})
    )


def run_media_command(command: list[str]) -> str:
    process = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    if process.returncode:
        raise RuntimeError(f"{Path(command[0]).name} 失败：{process.stderr[-3000:]}")
    return process.stdout


def probe_video(path: Path, ffprobe: str) -> dict:
    return json.loads(
        run_media_command(
            [
                ffprobe,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def embed_chapters(source: Path, output: Path, metadata: Path, ffmpeg: str) -> None:
    # Keep audio/video/subtitle streams. Old chapter data tracks are not media.
    # The new output's chapters replace the original chapter table, not its media.
    run_media_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(source),
            "-f",
            "ffmetadata",
            "-i",
            str(metadata),
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-map",
            "0:s?",
            "-map_metadata",
            "0",
            "-map_chapters",
            "1",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def verify_chapters(output: Path, chapters: list[dict], ffprobe: str) -> None:
    actual = probe_video(output, ffprobe).get("chapters", [])
    if len(actual) != len(chapters):
        raise RuntimeError("输出视频章节数量不匹配")
    for found, expected in zip(actual, chapters, strict=True):
        if abs(timestamp_ms(found["start_time"]) - expected["start_ms"]) > 1:
            raise RuntimeError("输出视频章节时间不匹配")
        if found.get("tags", {}).get("title") != expected["title"]:
            raise RuntimeError("输出视频章节标题不匹配")


def is_http_url(value: str) -> bool:
    parts = urlsplit(value)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def download_video(url: str, destination: Path, max_bytes: int) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        if int(response.headers.get("content-length", 0)) > max_bytes:
            raise ValueError("视频超过 --max-download-mb 限制")
        received = 0
        with destination.open("xb") as target:
            for chunk in response.iter_bytes(1024 * 1024):
                received += len(chunk)
                if received > max_bytes:
                    raise ValueError("视频超过 --max-download-mb 限制")
                target.write(chunk)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="视频 HTTP(S) URL；使用 --results 时也支持本地视频（支持 ~）")
    parser.add_argument(
        "--results", type=Path, help="已有 analyze_media JSON，跳过接口调用（支持 ~）"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="新建的输出目录，不允许覆盖已有目录（支持 ~）"
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="完整 analyze_media 接口 URL")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="采样间隔秒数，默认 1")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="最大距离阈值，默认 0.5；0 表示不筛选"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--timeout", type=float, default=1200, help="接口读取超时秒数，默认 1200；不会自动重试分析"
    )
    parser.add_argument("--max-download-mb", type=int, default=2048)
    return parser


def execute(args: argparse.Namespace) -> Path:
    for key in ("sample_interval", "timeout"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} 必须是大于 0 的有限数")
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        raise ValueError("threshold 必须在 0～1 之间")
    if not 1 <= args.top_k <= 10 or args.max_download_mb <= 0:
        raise ValueError("top-k 必须在 1～10 之间，max-download-mb 必须大于 0")
    remote = is_http_url(args.video)
    if not args.results and not remote:
        raise ValueError("接口仅接受 URL。本地视频请同时传入 --results，不会自动上传到其他服务")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("请先安装 ffmpeg 和 ffprobe，并加入 PATH")
    name = Path(unquote(urlsplit(args.video).path) if remote else args.video).stem or "video"
    safe_name = re.sub(r"[^\w.-]+", "_", name)[:80]
    output = (
        (
            args.output_dir
            or Path("data/media_reviews")
            / (safe_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        )
        .expanduser()
        .resolve()
    )
    # Fail closed rather than overwrite source videos or a previous report.
    output.mkdir(parents=True, exist_ok=False)
    print(f"输出目录：{output}", flush=True)
    with tempfile.TemporaryDirectory(prefix=".work-", dir=output) as work:
        if remote:
            print("下载视频（不修改远端原文件）…", flush=True)
            source = Path(work) / "source.video"
            download_video(args.video, source, args.max_download_mb * 1024 * 1024)
        else:
            source = Path(args.video).expanduser().resolve(strict=True)
        probe = probe_video(source, ffprobe)
        duration_ms = timestamp_ms(probe["format"]["duration"])
        if duration_ms <= 0 or not any(s["codec_type"] == "video" for s in probe["streams"]):
            raise ValueError("输入不是有效的有限时长视频")
        if args.results:
            payload = json.loads(args.results.expanduser().read_text(encoding="utf-8"))
        else:
            print("调用 analyze_media，视频较长时请等待；不会自动重复提交…", flush=True)
            with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=15)) as client:
                response = client.post(
                    args.api_url,
                    json={
                        "url": args.video,
                        "sample_interval": args.sample_interval,
                        "top_k": args.top_k,
                        "threshold": args.threshold,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        (output / "analysis.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        markers = normalize_results(payload, duration_ms)
        chapters = make_chapters(markers, duration_ms)
        metadata = output / "chapters.ffmetadata"
        metadata.write_text(metadata_text(chapters), encoding="utf-8")
        (output / "markers.json").write_text(
            json.dumps(
                {
                    "duration_ms": duration_ms,
                    "markers": markers,
                    "note": "人物区间表示连续采样命中，其余结果为单点；均需人工复核，无标记不代表内容安全。",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"共 {len(markers)} 条标记（单点/区间），正在无重编码写入章节…", flush=True)
        video = output / "marked.mp4"
        embed_chapters(source, video, metadata, ffmpeg)
        verify_chapters(video, chapters, ffprobe)
        (output / "review.html").write_text(
            render_review(name, markers, duration_ms), encoding="utf-8"
        )
        print(f"完成：{video}\n预览：{output / 'review.html'}", flush=True)
    return output


def main() -> int:
    try:
        execute(build_parser().parse_args())
    except (
        ValueError,
        OSError,
        RuntimeError,
        KeyError,
        httpx.HTTPError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
