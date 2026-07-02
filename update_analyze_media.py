import sys
from pathlib import Path
import re

file_path = Path('api/handlers.py')
content = file_path.read_text()

# Add _call_flags_analysis function
flags_func = """
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
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _sync_face_task(engine, frame, top_k, threshold, current_frame_time):
    all_results = []
    try:
        faces = engine.detect_faces(frame)
        for face_data in faces:
            face_img = face_data.get("face")
            if face_img is None: continue
            fa = face_data.get("facial_area", {})
            area = (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)
            conf = face_data.get("confidence") or 0
            frame_area = frame.shape[0] * frame.shape[1]
            if conf < 0.5 or area < MIN_FACE_PIXELS or area > frame_area * 0.8:
                continue
            _search_face_in_image(engine, face_img, None, top_k, threshold, all_results, current_frame_time)
        return all_results
    except Exception:
        return []
"""

# Now replace _process_analyze_media
analyze_media_new = """async def _process_analyze_media(url: str, sample_interval: float, top_k: int, threshold: float) -> list:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    engine = get_face_engine()
    
    async def _process_single_frame(frame, b64_img, current_frame_time):
        async def face_task():
            if frame is None:
                # for image mode, frame is already bytes, handled differently? 
                # actually for image mode, we can just use engine.search_faces but wait, we need to detect face first.
                return []
            return await run_in_inference_thread(_sync_face_task, engine, frame, top_k, threshold, current_frame_time)
            
        async def nsfw_task():
            visual_desc = await _call_qwen_image_analysis(b64_img)
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

        face_res, nsfw_res, ocr_res, flags_res = await asyncio.gather(
            face_task(),
            nsfw_task(),
            ocr_task(),
            flags_task()
        )
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
            frame_idx = 0
            
            tasks = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % int(max(fps * sample_interval, 1)) == 0:
                    current_frame_time = frame_idx / fps
                    _, buffer = cv2.imencode('.jpg', frame)
                    b64_img = base64.b64encode(buffer).decode('utf-8')
                    tasks.append(_process_single_frame(frame, b64_img, current_frame_time))
                frame_idx += 1
            cap.release()
            
            # Run all frames concurrently
            frame_results = await asyncio.gather(*tasks)
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        res = await _process_single_frame(frame, b64_img, 0.0)
        frame_results = [res]

    flattened_results = []
    
    # 验证人脸结果
    all_face_results = []
    for face_res, _, _, _, _ in frame_results:
        all_face_results.extend(face_res)
    verified_faces = await _verify_candidates(engine, all_face_results)
    
    # Group verified faces by frame time
    faces_by_time = {}
    for vf in verified_faces:
        t = vf.get("frame_time", 0.0)
        if t not in faces_by_time:
            faces_by_time[t] = []
        faces_by_time[t].append(vf)

    for _, nsfw_res, ocr_res, flags_res, ts in frame_results:
        formatted_ts = _format_timestamp(ts)
        
        face_list = faces_by_time.get(ts, [])
        for fr in face_list:
            flattened_results.append({
                "timestamp": formatted_ts,
                "category": "敏感人物",
                "text": fr.get("name", "未知人物")
            })
        
        if nsfw_res:
            flattened_results.append({
                "timestamp": formatted_ts,
                "category": nsfw_res["category"],
                "text": nsfw_res["text"]
            })
            
        if ocr_res:
            flattened_results.append({
                "timestamp": formatted_ts,
                "category": ocr_res["category"],
                "text": ocr_res["text"]
            })
            
        if flags_res:
            flattened_results.append({
                "timestamp": formatted_ts,
                "category": flags_res["category"],
                "text": flags_res["text"]
            })
            
    # Sort by timestamp
    flattened_results.sort(key=lambda x: x["timestamp"])
    
    return flattened_results
"""

# Find async def _process_analyze_media and replace up to async def _process_detect_nsfw
pattern = re.compile(r"async def _process_analyze_media.*?async def _process_detect_nsfw", re.DOTALL)

if not pattern.search(content):
    print("Could not find _process_analyze_media in handlers.py")
    sys.exit(1)

new_content = flags_func + "\n\n" + analyze_media_new + "\nasync def _process_detect_nsfw"
content = pattern.sub(new_content, content)
file_path.write_text(content)
print("Successfully updated handlers.py")
