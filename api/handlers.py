# pyright: ignore[reportUnusedFunction]

import asyncio
import base64
import logging
import os
import re
from pathlib import Path

import cv2
import httpx
import numpy as np

from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine

from .utils import (
    VIDEO_EXTENSIONS,
    VideoFrameSampler,
    _download_url_safe,
    _download_video_safe_sync,
    _extract_video_frames_for_ocr,
    _extract_video_windows,
)

_ANALYZE_MIN_FACE_PIXELS = 48
_logger = logging.getLogger(__name__)


class NsfwAnalysisError(RuntimeError):
    """A model failure must not be interpreted as a safe visual description."""


async def _search_video_frames(
    engine: FaceEngine,
    url: str,
    name: str | None,
    top_k: int,
    threshold: float,
    sample_interval: float,
    local_video_path: Path | None = None,
) -> tuple[int, list[dict]]:
    """Search faces from video by sampling frames."""
    if local_video_path is not None:
        video_path = local_video_path
        should_unlink = False
    else:
        video_path = Path(f"/tmp/ws_video_{os.urandom(8).hex()}.mp4")
        await asyncio.to_thread(
            _download_video_safe_sync,
            url,
            video_path,
            settings.max_file_size_mb * 100 * 1024 * 1024,
            timeout=900.0,
        )
        should_unlink = True

    try:
        all_results = []
        with VideoFrameSampler(video_path, sample_interval) as sampler:
            for window in sampler:
                frame = window[0].image
                current_frame_time = window[0].timestamp
                try:
                    results = await engine.search(
                        img_source=frame,
                        name=name,
                        top_k=top_k,
                        threshold=threshold,
                    )

                    for r in results:
                        r["frame_time"] = current_frame_time
                        # Include a JSON-safe crop when a matched-face bbox is available.
                        x, y, w, h = (
                            r.get("source_x"),
                            r.get("source_y"),
                            r.get("source_w"),
                            r.get("source_h"),
                        )
                        if x is not None and y is not None and w is not None and h is not None:
                            # ensure within frame bounds
                            y1, y2 = max(0, y), min(frame.shape[0], y + h)
                            x1, x2 = max(0, x), min(frame.shape[1], x + w)
                            if y2 > y1 and x2 > x1:
                                ok, encoded = cv2.imencode(".jpg", frame[y1:y2, x1:x2])
                                if ok:
                                    r["source_face_b64"] = base64.b64encode(encoded).decode("ascii")
                        all_results.append(r)
                except Exception as e:
                    print(f"Error searching frame: {e}")

            frame_idx = sampler.frames_read

        # Sort and dedupe results by distance
        all_results.sort(key=lambda x: x.get("distance", 1.0))
        seen = set()
        deduped = []
        for r in all_results:
            key = (r.get("name"), r.get("person_id"))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return frame_idx, deduped
    finally:
        if should_unlink and video_path.exists():
            video_path.unlink()


async def _call_ocr_api(base64_image: str) -> str:
    url = settings.model_api_url
    headers = {"Authorization": f"Bearer {settings.model_api_key}"}
    payload = {
        "model": "WasuAI/PaddleOCR-VL-1.6",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "提取图片中的所有文字。请只输出纯文本，绝对不要输出任何位置坐标（如<|LOC_0|>）、边界框或多余的解释。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            analysis = data["choices"][0]["message"]["content"].strip()

            # Clean up <|LOC_X|> bounding box tokens that the VLM might output
            analysis = re.sub(r"<\|LOC_\d+\|>", "", analysis)

            # Remove massive consecutive repetition (hallucinations like 王晓燕王晓燕...)
            analysis = re.sub(r"(.{1,30}?)\1{4,}", r"\1...", analysis)
            if len(analysis) > 500:
                analysis = analysis[:500] + "..."
            return analysis
        except Exception as e:
            if hasattr(e, "response") and e.response is not None:
                print(f"OCR API Error: {e.response.text}")
            else:
                print(f"OCR API Error: {e}")
            return ""


async def _call_llm_guard(text: str) -> dict:
    if not text.strip():
        return {"safe": True, "category": ""}

    url = settings.model_api_url
    headers = {"Authorization": f"Bearer {settings.model_api_key}"}
    payload = {
        "model": "WasuAI/WasuGuard-Gen-4B",
        "messages": [
            {
                "role": "system",
                "content": "你是一个内容安全审核员。请分析以下文本是否包含敏感、违规或不适宜的信息。如果完全安全，请输出：\nSafety: Safe\n\n如果不安全，请输出：\nSafety: Unsafe\nCategories: [具体的违规类别，例如：政治敏感、色情、辱骂、暴恐等，请务必用中文作答]",
            },
            {"role": "user", "content": text},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            analysis = data["choices"][0]["message"]["content"].strip()

            if "</think>" in analysis:
                analysis = analysis.split("</think>")[-1].strip()

            is_safe = True
            category = ""

            for line in analysis.split("\n"):
                line = line.strip()
                if "safety:" in line.lower() and "unsafe" in line.lower():
                    is_safe = False
                elif "categories:" in line.lower():
                    idx = line.lower().find("categories:")
                    category = line[idx + len("categories:") :].strip()

            if not is_safe and not category and "unsafe" in analysis.lower():
                # fallback extraction
                category = "未知敏感内容"

            # Map known English categories to Chinese
            guard_category_map = {
                "Violent": "血腥/暴力",
                "Non-violent Illegal Acts": "非暴力违法行为",
                "Sexual Content or Sexual Acts": "色情或性行为",
                "Personally Identifiable Information": "个人隐私信息",
                "Suicide & Self-Harm": "自杀与自残",
                "Unethical Acts": "不道德行为",
                "Politically Sensitive": "政治敏感内容",
                "Politically Sensitive Topics": "政治敏感内容",
                "Copyright Violation": "侵犯版权",
                "Jailbreak": "越狱/绕过安全限制",
            }

            # The model might output multiple categories comma-separated, or just one.
            # Handle possible partial matches or exact matches
            mapped_categories = set()
            for en_cat, cn_cat in guard_category_map.items():
                if en_cat.lower() in category.lower():
                    mapped_categories.add(cn_cat)

            if mapped_categories:
                category = "、".join(str(mc) for mc in mapped_categories)

            return {"safe": is_safe, "category": category}
        except Exception:
            return {"safe": True, "category": ""}

    url = settings.model_api_url
    headers = {"Authorization": f"Bearer {settings.model_api_key}"}
    payload = {
        "model": "WasuAI/WasuGuard-Gen-4B",
        "messages": [
            {
                "role": "system",
                "content": "你是一个内容安全审核员。请分析以下文本是否包含敏感、违规或不适宜的信息。如果完全安全，请输出：\nSafety: Safe\n\n如果不安全，请输出：\nSafety: Unsafe\nCategories: [具体的违规类别，例如：政治敏感、色情、辱骂、暴恐等，请务必用中文作答]",
            },
            {"role": "user", "content": text},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            analysis = data["choices"][0]["message"]["content"].strip()

            if "</think>" in analysis:
                analysis = analysis.split("</think>")[-1].strip()

            is_safe = True
            category = ""

            for line in analysis.split("\n"):
                line = line.strip()
                if line.lower().startswith("safety:"):
                    if "unsafe" in line.lower():
                        is_safe = False
                elif line.lower().startswith("categories:"):
                    category = line[len("categories:") :].strip()

            if "unsafe" in analysis.lower() and is_safe:
                is_safe = False
                category = analysis

            GUARD_CATEGORY_MAP = {
                "violent": "血腥/暴力",
                "non-violent illegal acts": "非暴力违法行为",
                "sexual content or sexual acts": "色情内容或性行为",
                "personally identifiable information": "个人身份信息",
                "suicide & self-harm": "自杀与自残",
                "unethical acts": "不道德行为",
                "politically sensitive topics": "政治敏感话题",
                "copyright violation": "侵犯版权",
                "jailbreak": "越狱",
            }

            lower_cat = category.lower()
            mapped_cats = []
            for en_key, cn_val in GUARD_CATEGORY_MAP.items():
                if en_key in lower_cat:
                    mapped_cats.append(cn_val)

            if mapped_cats:
                category = "、".join(mapped_cats)

            return {"safe": is_safe, "category": category}
        except Exception:
            return {"safe": True, "category": ""}


def _decode_nsfw_frame(image: str, max_width: int, max_height: int):
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode NSFW window frame")
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale < 1:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def _encode_nsfw_frame(frame) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Could not encode NSFW frame")
    return base64.b64encode(encoded).decode("ascii")


def _compose_nsfw_frames(images: list[str], timestamps: list[float] | None) -> tuple[str, str]:
    """A large target with smaller context thumbnails, never cropped."""
    if len(images) == 1:
        return images[0], "Only one target frame is provided."
    target = _decode_nsfw_frame(images[0], 1024, 1024)
    height, width = target.shape[:2]
    vertical = width >= height
    gap, header, border = 8, 36, 4
    thumb_width, thumb_height = max(1, (width - gap) // 2), max(1, (height - gap) // 2)
    frames = [target] + [
        _decode_nsfw_frame(image, thumb_width, thumb_height) for image in images[1:]
    ]
    panels = []
    for index, frame in enumerate(frames):
        panel = cv2.copyMakeBorder(
            frame, header, border, border, border, cv2.BORDER_CONSTANT, value=(32, 32, 32)
        )
        label = "TARGET 1" if index == 0 else f"CONTEXT {index + 1}"
        if timestamps is not None:
            label += f" | {timestamps[index]:.3f}s"
        cv2.putText(
            panel,
            label,
            (border + 4, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    # Landscape: target above a row of thumbnails. Portrait: target to the
    # left of a column of thumbnails. Headers stay outside the source images.
    context_width = (
        sum(p.shape[1] for p in panels[1:]) + gap * (len(panels) - 2)
        if vertical
        else max(p.shape[1] for p in panels[1:])
    )
    context_height = (
        max(p.shape[0] for p in panels[1:])
        if vertical
        else sum(p.shape[0] for p in panels[1:]) + gap * (len(panels) - 2)
    )
    sheet_width = (
        max(panels[0].shape[1], context_width)
        if vertical
        else panels[0].shape[1] + gap + context_width
    )
    sheet_height = (
        panels[0].shape[0] + gap + context_height
        if vertical
        else max(panels[0].shape[0], context_height)
    )
    sheet = np.full((sheet_height, sheet_width, 3), 32, dtype=np.uint8)
    h, w = panels[0].shape[:2]
    sheet[:h, :w] = panels[0]
    x, y = (0, h + gap) if vertical else (w + gap, 0)
    for panel in panels[1:]:
        ph, pw = panel.shape[:2]
        sheet[y : y + ph, x : x + pw] = panel
        if vertical:
            x += pw + gap
        else:
            y += ph + gap
    position = "above" if vertical else "on the left"
    return _encode_nsfw_frame(sheet), (
        f"The large TARGET 1 panel {position} is the frame to review. "
        f"The {len(images) - 1} smaller CONTEXT panels show later samples in numbered order. "
        "Labels, timestamps and borders are added annotations, not video content. "
    )


async def _request_nsfw_caption(
    client, image: str, prompt: str, *, context: str = "", max_tokens: int = 384
) -> str:
    payload = {
        "model": "WasuAI/JoyCaption",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                ],
            },
        ],
        "max_tokens": max_tokens,
    }
    if context:
        # Never pass the raw context caption to the verifier: JoyCaption can
        # copy later-frame facts even when explicitly asked not to. Only fixed
        # inspection topics cross this boundary; they assert no visible facts.
        focus = _nsfw_focus_questions(context)
        payload["messages"][-1]["content"][0]["text"] = focus + prompt
    original_prompt = payload["messages"][-1]["content"][0]["text"]
    for attempt, token_budget in enumerate((max_tokens, max(512, max_tokens * 2))):
        payload["max_tokens"] = token_budget
        if attempt:
            # Keep Chinese output and the original image/context boundaries
            # while asking for a concise, non-repeating caption on retry.
            payload["messages"][-1]["content"][0]["text"] = (
                original_prompt
                + " 请用简短中文描述，不要重复语句或罗列不存在的特征，描述完成后立即结束。"
            )
        response = await client.post(
            settings.model_api_url,
            headers={"Authorization": f"Bearer {settings.model_api_key}"},
            json=payload,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") != "length":
            break
        # Never send a partial caption to Guard, including after the retry.
        _logger.warning("NSFW caption truncated at max_tokens=%s", token_budget)
    if choice.get("finish_reason") == "length":
        raise ValueError("NSFW caption was truncated")
    analysis = choice["message"]["content"].split("</think>")[-1].strip()
    if not analysis:
        raise ValueError("NSFW caption was empty")
    return analysis


def _nsfw_focus_questions(context: str) -> str:
    topics = (
        (
            ("裸", "胸部", "生殖器", "私处", "nude", "nudity", "naked", "genital", "breast"),
            "clothing and exposed skin",
        ),
        (
            ("性行为", "性交", "口交", "自慰", "sexual", "penetrat", "masturbat"),
            "visible actions and body positions",
        ),
        (
            ("接触", "触碰", "拥抱", "亲吻", "搂抱", "kiss", "touch", "embrace"),
            "physical contact and interactions",
        ),
        (
            ("暴力", "血", "殴打", "武器", "枪", "刀", "violen", "blood", "weapon", "gun", "knife"),
            "visible injuries, objects and physical actions",
        ),
    )
    lower = context.lower()
    selected = [topic for keywords, topic in topics if any(word in lower for word in keywords)]
    if not selected:
        return ""
    return (
        "Inspection topics (questions only, not claims that anything is present): "
        + "; ".join(selected)
        + ". Verify against this image; ignore anything not visible. "
    )


async def _call_nsfw_analysis(
    b64_images: str | list[str], timestamps: list[float] | None = None
) -> str:
    images = [b64_images] if isinstance(b64_images, str) else b64_images
    if not 1 <= len(images) <= 3:
        raise ValueError("NSFW analysis requires one to three frames")
    if timestamps is not None and len(timestamps) != len(images):
        raise ValueError("Each frame must have its own timestamp")
    try:
        target = await asyncio.to_thread(_decode_nsfw_frame, images[0], 1024, 1024)
        target_image = await asyncio.to_thread(_encode_nsfw_frame, target)
        if len(images) > 1:
            sheet, layout = await asyncio.to_thread(_compose_nsfw_frames, images, timestamps)
    except Exception as exc:
        raise NsfwAnalysisError("NSFW 窗口图片拼接失败") from exc
    stage = "context" if len(images) > 1 else "target"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            context = ""
            if len(images) > 1:
                context = await _request_nsfw_caption(
                    client,
                    sheet,
                    layout
                    + (
                        "Write a brief factual caption of TARGET 1. Use CONTEXT only to clarify "
                        "actions already visible in TARGET. The samples have time gaps; a scene cut "
                        "breaks continuity. Do not transfer later-only content to TARGET or invent "
                        "missing frames. Focus on visible nudity, actions, physical contact or violence. "
                        "Avoid speculation about identity, age or hidden details. Output concise Chinese."
                    ),
                )
            prompt = (
                "Write a short descriptive caption for this image in Chinese, in one sentence. "
                "Describe only visible content in the attached image. "
                "Do not discuss notes, other images or absent objects."
            )
            # The downstream guard sees only this target-only verification.
            # No fallback to the context caption if verification fails.
            stage = "target"
            return await _request_nsfw_caption(
                client, target_image, prompt, context=context, max_tokens=160
            )
    except Exception as exc:
        _logger.exception(
            "NSFW model request or target verification failed: stage=%s timestamps=%s",
            stage,
            timestamps,
        )
        raise NsfwAnalysisError("NSFW 模型请求或目标帧复核失败，请检查模型服务") from exc


async def _process_detect_sensitive(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    unsafe_text_frames = []

    async def _analyze_text(timestamp, b64_img):
        text = await _call_ocr_api(b64_img)
        if text:
            guard = await _call_llm_guard(text)
            if not guard.get("safe", True):
                return {"timestamp": timestamp, "category": guard.get("category", ""), "text": text}
        return None

    if is_video:
        video_path = Path(f"/tmp/guard_video_{os.urandom(8).hex()}.mp4")
        try:
            await asyncio.to_thread(
                _download_video_safe_sync,
                url,
                video_path,
                settings.max_file_size_mb * 100 * 1024 * 1024,
            )
            frames_data = await asyncio.to_thread(
                _extract_video_frames_for_ocr, video_path, sample_interval
            )

            tasks = [_analyze_text(ts, b64) for ts, b64 in frames_data]
            results = await asyncio.gather(*tasks)
            for res in results:
                if res:
                    unsafe_text_frames.append(res)
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        res = await _analyze_text(None, b64_img)
        if res:
            res["type"] = "image"
            res.pop("timestamp", None)
            unsafe_text_frames.append(res)

    return {"unsafe_text_frames": unsafe_text_frames}


async def _call_flags_analysis(b64_img: str) -> str:
    url = settings.model_api_url
    headers = {"Authorization": f"Bearer {settings.model_api_key}"}
    payload = {
        "model": "WasuAI/WasuFlags3.5-4B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请检测图像中是否包含非法或政治团体的旗帜。如果不包含，请严格只输出一个字：无。如果包含，请描述是什么旗帜。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": 0.1,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            analysis = data["choices"][0]["message"]["content"].strip()
            if "</think>" in analysis:
                analysis = analysis.split("</think>")[-1].strip()
            return analysis
        except Exception:
            return "无"


def _format_timestamp(seconds: float) -> str:
    if seconds is None:
        return "00:00:00.000"
    total_ms = int(round(seconds * 1000))
    h, remainder = divmod(total_ms, 3600000)
    m, remainder = divmod(remainder, 60000)
    s, ms = divmod(remainder, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


async def _face_task(engine, frame, top_k, threshold, current_frame_time):
    all_results = []
    try:
        # Apply the same minimum during detection so the engine's default
        # 80px floor does not discard 48–79px query faces before this step.
        grouped = await engine.search_multi_face(
            img_source=frame,
            top_k=top_k,
            threshold=threshold,
            min_face_pixels=_ANALYZE_MIN_FACE_PIXELS,
        )
        for r in grouped["all_results"]:
            # source_* describes the enrolled database sample. Only the
            # query bbox belongs to the uploaded image/frame we are filtering
            # and cropping; never fall back to the sample's coordinates.
            bbox = r.get("query_face_bbox") or {}
            x, y, w, h = (bbox.get(key) for key in ("x", "y", "w", "h"))

            # Keep faces whose width and height are both at least 48px.
            if w is not None and h is not None and min(w, h) < _ANALYZE_MIN_FACE_PIXELS:
                continue

            r["timestamp"] = _format_timestamp(current_frame_time)
            r["frame_time"] = current_frame_time
            if x is not None and y is not None and w is not None and h is not None:
                y1, y2 = max(0, y), min(frame.shape[0], y + h)
                x1, x2 = max(0, x), min(frame.shape[1], x + w)
                if y2 > y1 and x2 > x1:
                    r["face_location"] = {
                        "x": x1 / frame.shape[1],
                        "y": y1 / frame.shape[0],
                        "w": (x2 - x1) / frame.shape[1],
                        "h": (y2 - y1) / frame.shape[0],
                    }
                    crop = frame[y1:y2, x1:x2]
                    ok, buf = cv2.imencode(".jpg", crop)
                    if ok:
                        r["face_image_b64"] = base64.b64encode(buf).decode("utf-8")
            all_results.append(r)
        return all_results
    except Exception:
        return []


def _merge_person_timelines(
    frame_results: list, sample_interval: float, sample_times: list[float] | None = None
) -> list[dict]:
    """Merge only face-task hits from consecutive sampled frames by category/name."""
    results = []
    previous = {}
    # Millisecond rounding must not split otherwise adjacent samples.
    max_gap = max(sample_interval, 0.0) + 0.001
    adjacent_samples = (
        set(zip(sample_times, sample_times[1:], strict=False)) if sample_times is not None else None
    )
    for face_res, _, _, _, ts in sorted(frame_results, key=lambda frame: frame[4]):
        current = {}
        formatted_ts = _format_timestamp(ts)
        for face in face_res:
            key = (face.get("category") or "敏感人物", face.get("name", "敏感人物"))
            location = face.get("face_location")
            sample = None
            if location:
                sample = {"time_ms": round(ts * 1000), "bbox": location}
                similarity = face.get("similarity")
                if similarity is not None:
                    sample["similarity"] = float(similarity)
            if key in current:
                row = current[key][2]
                if sample and sample not in row.setdefault("face_samples", []):
                    row["face_samples"].append(sample)
                continue
            prior = previous.get(key)
            adjacent = prior is not None and (
                (prior[1], ts) in adjacent_samples
                if adjacent_samples is not None
                else 0 <= ts - prior[1] <= max_gap
            )
            if adjacent:
                start, _, row = prior
                if formatted_ts != start:
                    row["timestamp"] = f"{start}~{formatted_ts}"
            else:
                start = formatted_ts
                row = {"timestamp": start, "category": key[0], "description": key[1]}
                results.append(row)
            if sample:
                row.setdefault("face_samples", []).append(sample)
            current[key] = (start, ts, row)
        # A sampled frame without this person terminates their current interval.
        previous = current
    return results


async def _process_analyze_media(
    url: str, sample_interval: float, top_k: int, threshold: float
) -> list:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    engine = get_face_engine()
    merge_interval = sample_interval
    sample_times = []

    async def _process_window(window):
        frame, b64_img, current_frame_time = window[0]

        async def face_task():
            if frame is None:
                return []
            return await _face_task(engine, frame, top_k, threshold, current_frame_time)

        async def nsfw_task():
            visual_desc = await _call_nsfw_analysis(
                [item[1] for item in window], [item[2] for item in window]
            )
            visual_guard = await _call_llm_guard(visual_desc)
            if not visual_guard.get("safe", True):
                return {"category": visual_guard.get("category", "视觉违规"), "text": visual_desc}
            return None

        async def ocr_task():
            text = await _call_ocr_api(b64_img)
            if text:
                text_guard = await _call_llm_guard(text)
                if not text_guard.get("safe", True):
                    return {"category": text_guard.get("category", "文本违规"), "text": text}
            return None

        async def flags_task():
            flags_desc = await _call_flags_analysis(b64_img)
            if (
                flags_desc
                and flags_desc != "无"
                and "不包含" not in flags_desc
                and "没有" not in flags_desc
            ):
                return {"category": "非法旗帜", "text": flags_desc}
            return None

        face_res, nsfw_res, ocr_res = await asyncio.gather(
            face_task(),
            nsfw_task(),
            ocr_task(),
            # flags_task()
        )
        flags_res = None
        return face_res, nsfw_res, ocr_res, flags_res, current_frame_time

    frame_results = []

    if is_video:
        video_path = Path(f"/tmp/analyze_video_{os.urandom(8).hex()}.mp4")
        try:
            await asyncio.to_thread(
                _download_video_safe_sync,
                url,
                video_path,
                settings.max_file_size_mb * 100 * 1024 * 1024,
                timeout=900.0,
            )

            queue = asyncio.Queue(maxsize=16)
            model_errors = []

            async def producer(sampler):
                for window in sampler:
                    sample_times.append(window[0].timestamp)
                    await queue.put(
                        tuple((frame.image, frame.b64, frame.timestamp) for frame in window)
                    )
                    await asyncio.sleep(0)

            async def consumer():
                while True:
                    item = await queue.get()
                    try:
                        res = await _process_window(item)
                        frame_results.append(res)
                    except asyncio.CancelledError:
                        raise
                    except NsfwAnalysisError as exc:
                        model_errors.append(exc)
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                    finally:
                        queue.task_done()

            NUM_CONSUMERS = queue.maxsize // 2
            consumers = [asyncio.create_task(consumer()) for _ in range(NUM_CONSUMERS)]

            try:
                with VideoFrameSampler(video_path, sample_interval, max_dimension=1080) as sampler:
                    merge_interval = sampler.interval
                    await producer(sampler)
                await queue.join()
                if model_errors:
                    raise model_errors[0]
            finally:
                for consumer_task in consumers:
                    consumer_task.cancel()
                await asyncio.gather(*consumers, return_exceptions=True)
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Downsample single image for VLM
        h, w = frame.shape[:2]
        if max(h, w) > 1080:
            scale = 1080 / max(h, w)
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small_frame = frame

        _, buffer = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_img = base64.b64encode(buffer).decode("utf-8")

        res = await _process_window(((frame, b64_img, 0.0),))
        frame_results = [res]

    flattened_results = _merge_person_timelines(
        frame_results, merge_interval, sample_times if is_video else None
    )

    for _, nsfw_res, ocr_res, flags_res, ts in frame_results:
        formatted_ts = _format_timestamp(ts)

        if nsfw_res:
            flattened_results.append(
                {
                    "timestamp": formatted_ts,
                    "category": nsfw_res["category"],
                    "description": nsfw_res["text"],
                }
            )

        if ocr_res:
            flattened_results.append(
                {
                    "timestamp": formatted_ts,
                    "category": ocr_res["category"],
                    "description": ocr_res["text"],
                }
            )

        if flags_res:
            flattened_results.append(
                {
                    "timestamp": formatted_ts,
                    "category": flags_res["category"],
                    "description": flags_res["text"],
                }
            )

    # Sort by timestamp
    flattened_results.sort(key=lambda x: x["timestamp"].split("~", 1)[0])

    return flattened_results


async def _process_detect_nsfw(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

    unsafe_text_frames = []
    nsfw_visual_results = []

    async def _analyze_frame(window):
        timestamp, b64_img = window[0]
        times = [item[0] for item in window] if timestamp is not None else None
        visual_desc = await _call_nsfw_analysis([item[1] for item in window], times)

        # Use LLM guard to evaluate the visual description
        visual_guard = await _call_llm_guard(visual_desc)
        is_nsfw = not visual_guard.get("safe", True)
        if is_nsfw:
            cat = visual_guard.get("category", "违规")
            if cat:
                visual_desc = f"[{cat}] {visual_desc}"

        text = await _call_ocr_api(b64_img)
        text_guard = None
        if text:
            text_guard = await _call_llm_guard(text)

        return timestamp, is_nsfw, visual_desc, text_guard, text

    if is_video:
        video_path = Path(f"/tmp/nsfw_video_{os.urandom(8).hex()}.mp4")
        try:
            await asyncio.to_thread(
                _download_video_safe_sync,
                url,
                video_path,
                settings.max_file_size_mb * 100 * 1024 * 1024,
            )
            frames_data = await asyncio.to_thread(
                _extract_video_windows, video_path, sample_interval
            )

            tasks = [_analyze_frame(window) for window in frames_data]
            frame_results = await asyncio.gather(*tasks)

            for timestamp, is_nsfw, visual_desc, text_guard, frame_text in frame_results:
                if is_nsfw:
                    nsfw_visual_results.append(
                        {"timestamp": timestamp, "confidence": 1.0, "description": visual_desc}
                    )

                if text_guard and not text_guard.get("safe", True):
                    unsafe_text_frames.append(
                        {
                            "timestamp": timestamp,
                            "category": text_guard.get("category", ""),
                            "text": frame_text,
                        }
                    )
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        _, is_nsfw, visual_desc, text_guard, frame_text = await _analyze_frame(((None, b64_img),))
        if is_nsfw:
            nsfw_visual_results.append(
                {"type": "image", "confidence": 1.0, "description": visual_desc}
            )

        if text_guard and not text_guard.get("safe", True):
            unsafe_text_frames.append(
                {"type": "image", "category": text_guard.get("category", ""), "text": frame_text}
            )

    return {"visual_analysis": nsfw_visual_results, "unsafe_text_frames": unsafe_text_frames}
