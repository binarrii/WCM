# pyright: ignore[reportUnusedFunction]

import asyncio
import base64
import os
from pathlib import Path
import cv2
import httpx
import numpy as np

from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine
from .utils import (
    _download_url_safe, 
    _download_video_safe_sync, 
    _extract_video_frames_for_ocr,
    VIDEO_EXTENSIONS
)


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
        _download_video_safe_sync(url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024, timeout=900.0)
        should_unlink = True

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        all_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % int(fps * sample_interval) == 0:
                current_frame_time = frame_idx / fps if fps > 0 else 0
                try:
                    # Send entire frame to DeepFace /search API directly
                    results = await asyncio.to_thread(
                        engine.search,
                        img_source=frame,
                        name=name,
                        top_k=top_k,
                        threshold=threshold,
                    )
                    
                    for r in results:
                        r["frame_time"] = current_frame_time
                        # Try to extract the cropped face from the bounding box if provided by DeepFace
                        x, y, w, h = r.get("source_x"), r.get("source_y"), r.get("source_w"), r.get("source_h")
                        if x is not None and y is not None and w is not None and h is not None:
                            # ensure within frame bounds
                            y1, y2 = max(0, y), min(frame.shape[0], y + h)
                            x1, x2 = max(0, x), min(frame.shape[1], x + w)
                            if y2 > y1 and x2 > x1:
                                r["source_face"] = frame[y1:y2, x1:x2]
                        all_results.append(r)
                except Exception as e:
                    print(f"Error searching frame: {e}")
                    pass

            frame_idx += 1

        cap.release()

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
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/PaddleOCR-VL-1.6",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "提取图片中的所有文字。请只输出纯文本，绝对不要输出任何位置坐标（如<|LOC_0|>）、边界框或多余的解释。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            analysis = data["choices"][0]["message"]["content"].strip()
            
            import re
            # Clean up <|LOC_X|> bounding box tokens that the VLM might output
            analysis = re.sub(r'<\|LOC_\d+\|>', '', analysis)
            
            # Remove massive consecutive repetition (hallucinations like 王晓燕王晓燕...)
            analysis = re.sub(r'(.{1,30}?)\1{4,}', r'\1...', analysis)
            if len(analysis) > 500:
                analysis = analysis[:500] + "..."
            return analysis
        except Exception as e:
            if hasattr(e, 'response') and getattr(e, 'response') is not None:
                print(f"OCR API Error: {e.response.text}")
            else:
                print(f"OCR API Error: {e}")
            return ""


async def _call_llm_guard(text: str) -> dict:
    if not text.strip():
        return {"safe": True, "category": ""}
    
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/WasuGuard-Gen-4B",
        "messages": [
            {
                "role": "system",
                "content": "你是一个内容安全审核员。请分析以下文本是否包含敏感、违规或不适宜的信息。如果完全安全，请输出：\nSafety: Safe\n\n如果不安全，请输出：\nSafety: Unsafe\nCategories: [具体的违规类别，例如：政治敏感、色情、辱骂、暴恐等，请务必用中文作答]"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1
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
                    category = line[idx + len("categories:"):].strip()
                    
            if not is_safe and not category:
                if "unsafe" in analysis.lower():
                    # fallback extraction
                    category = "未知敏感内容"
                    
            # Map known English categories to Chinese
            guard_category_map = {
                "Violent": "暴力",
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
            mapped_categories = []
            for en_cat, cn_cat in guard_category_map.items():
                if en_cat.lower() in category.lower():
                    mapped_categories.append(cn_cat)
                    
            if mapped_categories:
                category = "、".join(mapped_categories)
                    
            return {"safe": is_safe, "category": category}
        except Exception as e:
            return {"safe": True, "category": ""}
    
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/WasuGuard-Gen-4B",
        "messages": [
            {
                "role": "system",
                "content": "你是一个内容安全审核员。请分析以下文本是否包含敏感、违规或不适宜的信息。如果完全安全，请输出：\nSafety: Safe\n\n如果不安全，请输出：\nSafety: Unsafe\nCategories: [具体的违规类别，例如：政治敏感、色情、辱骂、暴恐等，请务必用中文作答]"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1
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
                    category = line[len("categories:"):].strip()
                    
            if "unsafe" in analysis.lower() and is_safe:
                is_safe = False
                category = analysis
                
            GUARD_CATEGORY_MAP = {
                "violent": "暴力",
                "non-violent illegal acts": "非暴力违法行为",
                "sexual content or sexual acts": "色情内容或性行为",
                "personally identifiable information": "个人身份信息",
                "suicide & self-harm": "自杀与自残",
                "unethical acts": "不道德行为",
                "politically sensitive topics": "政治敏感话题",
                "copyright violation": "侵犯版权",
                "jailbreak": "越狱"
            }
            
            lower_cat = category.lower()
            mapped_cats = []
            for en_key, cn_val in GUARD_CATEGORY_MAP.items():
                if en_key in lower_cat:
                    mapped_cats.append(cn_val)
                    
            if mapped_cats:
                category = "、".join(mapped_cats)
                
            return {"safe": is_safe, "category": category}
        except Exception as e:
            return {"safe": True, "category": ""}

async def _call_nsfw_analysis(b64_img: str) -> str:
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/JoyCaption",
        "messages": [
            {"role": "system", "content": "你是一位专业的看图配文助手，擅长使用中文精细描述图片内容！"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述这张图片的内容。如果画面中包含裸露、色情暗示、血腥暴力等违规内容，请务必详细描述出来。如果不包含违规内容，请客观描述画面内容即可。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.3,
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
        except Exception as e:
            return f"无法获取描述: {e}"

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
                _download_video_safe_sync, url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024
            )
            frames_data = await asyncio.to_thread(_extract_video_frames_for_ocr, video_path, sample_interval)
            
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
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        res = await _analyze_text(None, b64_img)
        if res:
            res["type"] = "image"
            res.pop("timestamp", None)
            unsafe_text_frames.append(res)

    return {
        "unsafe_text_frames": unsafe_text_frames
    }




async def _call_flags_analysis(b64_img: str) -> str:
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/WasuFlags3.5-4B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请检测图像中是否包含非法或政治团体的旗帜。如果不包含，请严格只输出一个字：无。如果包含，请描述是什么旗帜。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
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
        except Exception as e:
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
        results = await engine.search(
            img_source=frame,
            top_k=top_k,
            threshold=threshold
        )
        for r in results:
            r["timestamp"] = _format_timestamp(current_frame_time)
            r["frame_time"] = current_frame_time
            x, y, w, h = r.get("source_x"), r.get("source_y"), r.get("source_w"), r.get("source_h")
            if x is not None and y is not None and w is not None and h is not None:
                y1, y2 = max(0, y), min(frame.shape[0], y + h)
                x1, x2 = max(0, x), min(frame.shape[1], x + w)
                if y2 > y1 and x2 > x1:
                    crop = frame[y1:y2, x1:x2]
                    _, buf = cv2.imencode('.jpg', crop)
                    r["face_image_b64"] = base64.b64encode(buf).decode('utf-8')
            all_results.append(r)
        return all_results
    except Exception:
        return []


async def _process_analyze_media(url: str, sample_interval: float, top_k: int, threshold: float) -> list:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    engine = get_face_engine()
    
    async def _process_single_frame(frame, b64_img, current_frame_time):
        async def face_task():
            if frame is None:
                return []
            return await _face_task(engine, frame, top_k, threshold, current_frame_time)
            
        async def nsfw_task():
            visual_desc = await _call_nsfw_analysis(b64_img)
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
            if flags_desc and flags_desc != "无" and "不包含" not in flags_desc and "没有" not in flags_desc:
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
                _download_video_safe_sync, url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024, timeout=900.0
            )
            
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 25.0
            
            queue = asyncio.Queue(maxsize=16)
            
            async def producer():
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % int(max(fps * sample_interval, 1)) == 0:
                        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        current_frame_time = msec / 1000.0 if msec >= 0 else frame_idx / fps
                        # Downsample frame for VLM to reduce payload size and processing time
                        # Keep max dimension at 1080 to preserve OCR readability while saving bandwidth
                        h, w = frame.shape[:2]
                        if max(h, w) > 1080:
                            scale = 1080 / max(h, w)
                            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                        else:
                            small_frame = frame
                            
                        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        b64_img = base64.b64encode(buffer).decode('utf-8')
                        await queue.put((small_frame, b64_img, current_frame_time))
                    frame_idx += 1
                    
                    if frame_idx % (queue.maxsize * 2) == 0:
                        await asyncio.sleep(0)
                        
                cap.release()
                
            async def consumer():
                while True:
                    item = await queue.get()
                    try:
                        frame, b64_img, current_frame_time = item
                        res = await _process_single_frame(frame, b64_img, current_frame_time)
                        frame_results.append(res)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                    finally:
                        queue.task_done()

            NUM_CONSUMERS = queue.maxsize // 2
            consumers = [asyncio.create_task(consumer()) for _ in range(NUM_CONSUMERS)]
            
            await producer()
            await queue.join()
            
            for c in consumers:
                c.cancel()
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
            
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        res = await _process_single_frame(frame, b64_img, 0.0)
        frame_results = [res]

    flattened_results = []
    
    # 验证人脸结果
    all_face_results = []
    for face_res, _, _, _, _ in frame_results:
        all_face_results.extend(face_res)
    verified_faces = all_face_results
    
    # Group verified faces by frame time
    faces_by_time = {}
    for vf in verified_faces:
        t = vf.get("frame_time", 0.0) or vf.get("timestamp", 0.0)
        if t not in faces_by_time:
            faces_by_time[t] = []
        faces_by_time[t].append(vf)

    for _, nsfw_res, ocr_res, flags_res, ts in frame_results:
        formatted_ts = _format_timestamp(ts)
        
        face_list = faces_by_time.get(ts, []) or faces_by_time.get(formatted_ts, [])
        for fr in face_list:
            flattened_results.append({
                "timestamp": formatted_ts,
                "category": fr.get("category", None) or "敏感人物",
                "description": fr.get("name", "敏感人物")
            })
        
        if nsfw_res:
            flattened_results.append({
                "timestamp": formatted_ts,
                "category": nsfw_res["category"],
                "description": nsfw_res["text"]
            })
            
        if ocr_res:
            flattened_results.append({
                "timestamp": formatted_ts,
                "category": ocr_res["category"],
                "description": ocr_res["text"]
            })
            
        if flags_res:
            flattened_results.append({
                "timestamp": formatted_ts,
                "category": flags_res["category"],
                "description": flags_res["text"]
            })
            
    # Sort by timestamp
    flattened_results.sort(key=lambda x: x["timestamp"])
    
    return flattened_results

async def _process_detect_nsfw(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    
    unsafe_text_frames = []
    nsfw_visual_results = []
    
    async def _analyze_frame(timestamp, b64_img):
        visual_desc = await _call_nsfw_analysis(b64_img)
        
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
                _download_video_safe_sync, url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024
            )
            frames_data = await asyncio.to_thread(_extract_video_frames_for_ocr, video_path, sample_interval)
            
            tasks = [_analyze_frame(ts, b64) for ts, b64 in frames_data]
            frame_results = await asyncio.gather(*tasks)
            
            for timestamp, is_nsfw, visual_desc, text_guard, frame_text in frame_results:
                if is_nsfw:
                    nsfw_visual_results.append({
                        "timestamp": timestamp,
                        "confidence": 1.0,
                        "description": visual_desc
                    })
                
                if text_guard and not text_guard.get("safe", True):
                    unsafe_text_frames.append({
                        "timestamp": timestamp,
                        "category": text_guard.get("category", ""),
                        "text": frame_text
                    })
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        _, is_nsfw, visual_desc, text_guard, frame_text = await _analyze_frame(None, b64_img)
        if is_nsfw:
            nsfw_visual_results.append({
                "type": "image",
                "confidence": 1.0,
                "description": visual_desc
            })
                
        if text_guard and not text_guard.get("safe", True):
            unsafe_text_frames.append({
                "type": "image",
                "category": text_guard.get("category", ""),
                "text": frame_text
            })

    return {
        "visual_analysis": nsfw_visual_results,
        "unsafe_text_frames": unsafe_text_frames
    }



