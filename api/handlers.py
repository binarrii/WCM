# pyright: ignore[reportUnusedFunction]

import asyncio
import base64
import json
import os
import uuid
import io
from pathlib import Path
from typing import Union
import cv2
import httpx
import numpy as np
from PIL import Image

from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine
from .utils import (
    run_in_inference_thread, 
    _download_url_safe, 
    _download_video_safe_sync, 
    _extract_video_frames_for_ocr,
    VIDEO_EXTENSIONS,
    MIN_FACE_PIXELS
)

nsfw_pipeline = None

async def _verify_candidates(engine: FaceEngine, candidates: list[dict], default_source_img: Union[bytes, np.ndarray, str] = None) -> list[dict]:
    """Verify vector search candidates with DeepFace.verify."""
    verified_results = []
    for cand in candidates:
        source_img = cand.pop("source_face", default_source_img)
        if source_img is None:
            continue

        # source_img is a cropped face from detect_faces (float64 [0,1]); convert
        # to uint8 [0,255] for the same reason as db_face below — see
        # _crop_largest_face. db images are read fresh as uint8 already.
        if isinstance(source_img, np.ndarray) and source_img.dtype != np.uint8:
            source_img = (np.clip(source_img, 0, 1) * 255).astype(np.uint8)

        file_path = cand.get("file_path")
        file_url = cand.get("file_url")

        db_img = None
        try:
            if file_path and os.path.exists(file_path):
                db_img = cv2.imread(file_path, cv2.IMREAD_COLOR_BGR)
            elif file_url:
                img_bytes = await _download_url_safe(file_url, settings.max_file_size_mb * 1024 * 1024, timeout=10.0)
                nparr = np.frombuffer(img_bytes, np.uint8)
                db_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)
        except Exception:
            pass

        if db_img is None:
            # "if no file path (or url, or invalid) skip this step"
            continue

        # Pre-crop the DB image's face so verify_faces receives two already-cropped
        # faces. verify_faces still uses retinaface+align, which is well-behaved on
        # cropped faces (it skips re-detection and treats them as-is). Without this
        # step, passing a full DB image to verify produces garbage embeddings.
        db_face = await run_in_inference_thread(_crop_largest_face, engine, db_img)
        if db_face is None:
            continue

        verified = await run_in_inference_thread(engine.verify_faces, source_img, db_face)
        if verified:
            cand["verified"] = True
            verified_results.append(cand)

    return verified_results


def _crop_largest_face(engine: FaceEngine, img: np.ndarray) -> np.ndarray | None:
    """Detect faces in ``img`` and return the largest cropped face, or None.

    Reuses ``FaceEngine.detect_faces`` (the same crop function used by
    register/search) so the crop is consistent with the embedding pipeline.

    The returned face is converted to uint8 [0,255]. DeepFace.extract_faces
    yields float64 [0,1] crops, but DeepFace.verify's preprocessing assumes
    uint8 [0,255]; feeding it float64 [0,1] collapses the embedding so that
    any two faces score distance ~0, making verification meaningless.
    """
    faces = engine.detect_faces(img)
    if not faces:
        return None

    def get_face_area(f):
        fa = f.get("facial_area", {})
        return (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)

    face = max(faces, key=get_face_area).get("face")
    if face is None:
        return None
    if face.dtype != np.uint8:
        face = (np.clip(face, 0, 1) * 255).astype(np.uint8)
    return face


async def _detect_and_crop_face(engine: FaceEngine, url: str) -> dict | None:
    """Download image, detect face, crop and return face embedding (in-memory)."""
    try:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024, timeout=30.0)
        return _detect_and_crop_face_from_bytes(engine, img_bytes)
    except Exception:
        return None


def _detect_and_crop_face_from_bytes(engine: FaceEngine, img_bytes: bytes) -> dict | None:
    """Detect face from image bytes, crop and return face embedding (in-memory).

    Returns dict with 'embedding' and 'confidence' if face found, None otherwise.
    """
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)
        if img_array is None:
            return None

        faces = engine.detect_faces(img_array)
        if not faces:
            return None

        def get_face_area(f):
            fa = f.get("facial_area", {})
            return (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)
        best_face = max(faces, key=get_face_area)
        face_img = best_face.get("face")
        confidence = best_face.get("confidence", 0.0)

        if face_img is None or confidence < 0.6:
            return None

        embedding = engine.generate_embedding(face_img)
        return {
            "embedding": embedding,
            "confidence": confidence,
            "face": face_img,
        }
    except Exception:
        return None


def _search_video_frames(
    engine: FaceEngine,
    url: str,
    name: str | None,
    top_k: int,
    threshold: float,
    sample_interval: float,
) -> tuple[int, list[dict]]:
    """Search faces from video by sampling frames."""
    video_path = Path(f"/tmp/ws_video_{os.urandom(8).hex()}.mp4")
    try:
        _download_video_safe_sync(url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024, timeout=900.0)

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
                    faces = engine.detect_faces(frame)
                    for face_data in faces:
                        face_img = face_data.get("face")
                        if face_img is None:
                            continue
                        # Filter out bad detections
                        fa = face_data.get("facial_area", {})
                        area = (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)
                        conf = face_data.get("confidence") or 0
                        frame_area = frame.shape[0] * frame.shape[1]
                        # Skip if confidence is very low or face covers most of frame (detection failed)
                        if conf < 0.5 or area < MIN_FACE_PIXELS or area > frame_area * 0.8:
                            continue

                        # # --- DEBUG: save cropped face to temp dir ---
                        # import tempfile as _debug_tmp
                        # _debug_dir = _debug_tmp.mkdtemp(prefix="debug/face_search_debug_")
                        # _debug_fname = _debug_dir + f"/frame{frame_idx:06d}_t{current_frame_time:.2f}_conf{conf:.2f}_area{area}.png"
                        # _debug_face = (np.clip(face_img, 0, 1) * 255).astype(np.uint8) if face_img.dtype != np.uint8 else face_img
                        # cv2.imwrite(_debug_fname, _debug_face)
                        # print(f"[DEBUG FACE] frame={frame_idx} time={current_frame_time:.2f}s conf={conf:.2f} area={area} saved={_debug_fname}")
                        # # --- END DEBUG ---

                        _search_face_in_image(engine, face_img, name, top_k, threshold, all_results, current_frame_time)
                except Exception:
                    pass  # Skip frames that fail detection

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
        # return frame_idx, deduped[:top_k]
    finally:
        if video_path.exists():
            video_path.unlink()


def _search_face_in_image(
    engine: FaceEngine,
    face_img,
    name: str | None,
    top_k: int,
    threshold: float,
    all_results: list,
    frame_time: float | None = None,
):
    """Search a single face from a frame (in-memory, no temp files)."""
    try:
        embedding = engine.generate_embedding(face_img)
        results = engine.search(
            embedding=embedding,
            name=name,
            top_k=top_k,
            threshold=threshold,
        )
        for r in results:
            r["frame_time"] = frame_time
            r["source_face"] = face_img
        all_results.extend(results)
    except Exception:
        pass  # Skip faces that fail embedding generation


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
                    {"type": "text", "text": "Extract all text from this image. Output only the extracted text."},
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
            return data["choices"][0]["message"]["content"].strip()
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
                if line.lower().startswith("safety:"):
                    if "unsafe" in line.lower():
                        is_safe = False
                elif line.lower().startswith("categories:"):
                    category = line[len("categories:"):].strip()
                    
            if "unsafe" in analysis.lower() and is_safe:
                is_safe = False
                category = analysis
                
            return {"safe": is_safe, "category": category}
        except Exception as e:
            return {"safe": True, "category": ""}

async def _call_qwen_image_analysis(b64_img: str) -> str:
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/Qwen3.6-27B-Abliterated",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是一张可能包含违规内容的图片。请用非常简短的一句话（中文）描述该图片属于哪类违规内容（例如：裸露、血腥暴力、色情暗示等），只输出简短描述，不要输出任何其他内容。如果完全正常，请输出：正常。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "max_tokens": 64,
        "temperature": 0.3
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
            return "无法获取描述"

async def _process_detect_sensitive(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    unsafe_text_frames = []
    
    async def _analyze_text(timestamp, b64_img):
        text = await _call_ocr_api(b64_img)
        if text:
            guard = await _call_llm_guard(text)
            if not guard.get("safe", True):
                return {"timestamp": timestamp, "category": guard.get("category", "")}
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



def get_nsfw_pipeline():
    global nsfw_pipeline
    if nsfw_pipeline is None:
        import torch
        from transformers import pipeline
        device = -1 # Force CPU to avoid CUDA conflicts with TensorFlow
        nsfw_pipeline = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return nsfw_pipeline


def detect_visual_nsfw(image_bytes: bytes) -> list:
    image = Image.open(io.BytesIO(image_bytes))
    pipe = get_nsfw_pipeline()
    return pipe(image)


async def _process_analyze_media(url: str, sample_interval: float, top_k: int, threshold: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    engine = get_face_engine()
    
    unsafe_text_frames = []
    nsfw_visual_results = []
    face_search_results = []
    
    async def _analyze_frame(timestamp, b64_img):
        img_bytes = base64.b64decode(b64_img)
        visual_result = await run_in_inference_thread(detect_visual_nsfw, img_bytes)
        
        visual_desc = None
        for label_score in visual_result:
            if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                visual_desc = await _call_qwen_image_analysis(b64_img)
                break
                
        text = await _call_ocr_api(b64_img)
        text_guard = None
        if text:
            text_guard = await _call_llm_guard(text)
            
        return timestamp, visual_result, visual_desc, text_guard

    if is_video:
        frames_processed, face_results = await run_in_inference_thread(
            _search_video_frames,
            engine, url, None, max(min(top_k, 10), 1),
            max(min(threshold, 1.0), 0.0), sample_interval
        )
        face_search_results = await _verify_candidates(engine, face_results)
        
        video_path = Path(f"/tmp/analyze_video_{os.urandom(8).hex()}.mp4")
        try:
            await asyncio.to_thread(
                _download_video_safe_sync, url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024
            )
            frames_data = await asyncio.to_thread(_extract_video_frames_for_ocr, video_path, sample_interval)
            
            tasks = [_analyze_frame(ts, b64) for ts, b64 in frames_data]
            frame_results = await asyncio.gather(*tasks)
            
            for timestamp, visual_result, visual_desc, text_guard in frame_results:
                for label_score in visual_result:
                    if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                        nsfw_visual_results.append({
                            "timestamp": timestamp,
                            "confidence": label_score["score"],
                            "description": visual_desc
                        })
                
                if text_guard and not text_guard.get("safe", True):
                    unsafe_text_frames.append({
                        "timestamp": timestamp,
                        "category": text_guard.get("category", "")
                    })
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        face_result = await run_in_inference_thread(_detect_and_crop_face_from_bytes, engine, img_bytes)
        if face_result and "embedding" in face_result:
            search_res = await run_in_inference_thread(
                engine.search_faces, face_result["embedding"], top_k, threshold
            )
            face_search_results = await _verify_candidates(engine, search_res)

        _, visual_result, visual_desc, text_guard = await _analyze_frame(None, b64_img)
        for label_score in visual_result:
            if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                nsfw_visual_results.append({
                    "type": "image",
                    "confidence": label_score["score"],
                    "description": visual_desc
                })
                
        if text_guard and not text_guard.get("safe", True):
            unsafe_text_frames.append({
                "type": "image",
                "category": text_guard.get("category", "")
            })

    return {
        "face_search_results": face_search_results,
        "is_nsfw": len(nsfw_visual_results) > 0,
        "visual_analysis": nsfw_visual_results,
        "unsafe_text_frames": unsafe_text_frames
    }

async def _process_detect_nsfw(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    
    unsafe_text_frames = []
    nsfw_visual_results = []
    
    async def _analyze_frame(timestamp, b64_img):
        img_bytes = base64.b64decode(b64_img)
        visual_result = await run_in_inference_thread(detect_visual_nsfw, img_bytes)
        
        visual_desc = None
        for label_score in visual_result:
            if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                visual_desc = await _call_qwen_image_analysis(b64_img)
                break
                
        text = await _call_ocr_api(b64_img)
        text_guard = None
        if text:
            text_guard = await _call_llm_guard(text)
            
        return timestamp, visual_result, visual_desc, text_guard
        
    if is_video:
        video_path = Path(f"/tmp/nsfw_video_{os.urandom(8).hex()}.mp4")
        try:
            await asyncio.to_thread(
                _download_video_safe_sync, url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024
            )
            frames_data = await asyncio.to_thread(_extract_video_frames_for_ocr, video_path, sample_interval)
            
            tasks = [_analyze_frame(ts, b64) for ts, b64 in frames_data]
            frame_results = await asyncio.gather(*tasks)
            
            for timestamp, visual_result, visual_desc, text_guard in frame_results:
                for label_score in visual_result:
                    if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                        nsfw_visual_results.append({
                            "timestamp": timestamp,
                            "confidence": label_score["score"],
                            "description": visual_desc
                        })
                
                if text_guard and not text_guard.get("safe", True):
                    unsafe_text_frames.append({
                        "timestamp": timestamp,
                        "category": text_guard.get("category", "")
                    })
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        _, visual_result, visual_desc, text_guard = await _analyze_frame(None, b64_img)
        for label_score in visual_result:
            if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                nsfw_visual_results.append({
                    "type": "image",
                    "confidence": label_score["score"],
                    "description": visual_desc
                })
                
        if text_guard and not text_guard.get("safe", True):
            unsafe_text_frames.append({
                "type": "image",
                "category": text_guard.get("category", "")
            })

    return {
        "is_nsfw": len(nsfw_visual_results) > 0,
        "visual_analysis": nsfw_visual_results,
        "unsafe_text_frames": unsafe_text_frames
    }



