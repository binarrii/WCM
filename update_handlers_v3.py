import re

with open('api/handlers.py', 'r') as f:
    content = f.read()

# Replace _call_llm_guard and _call_qwen_nsfw with the new _call_llm_guard and add _call_qwen_image_analysis
old_llm_funcs = re.search(r'async def _call_llm_guard.*?async def _process_detect_sensitive', content, re.DOTALL).group(0)
new_llm_funcs = """async def _call_llm_guard(text: str) -> dict:
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
                "content": "你是一个内容安全审核员。请分析以下文本是否包含敏感、违规或不适宜的信息。如果完全安全，请输出：\\nSafety: Safe\\n\\n如果不安全，请输出：\\nSafety: Unsafe\\nCategories: [具体的违规类别，例如：政治敏感、色情、辱骂、暴恐等，请务必用中文作答]"
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
            
            for line in analysis.split("\\n"):
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

async def _process_detect_sensitive"""
content = content.replace(old_llm_funcs, new_llm_funcs)

# Update _process_detect_sensitive
old_sensitive = re.search(r'async def _process_detect_sensitive.*?return \{(.*?)\}', content, re.DOTALL).group(0)
new_sensitive = """async def _process_detect_sensitive(url: str, sample_interval: float) -> dict:
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
    }"""
content = content.replace(old_sensitive, new_sensitive)

# Remove _call_qwen_nsfw completely if it exists (it's between detect_visual_nsfw and _process_analyze_media)
qwen_func_match = re.search(r'async def _call_qwen_nsfw.*?async def _process_analyze_media', content, re.DOTALL)
if qwen_func_match:
    content = content.replace(qwen_func_match.group(0), 'async def _process_analyze_media')

# Update _process_analyze_media
old_analyze = re.search(r'async def _process_analyze_media.*?return \{(.*?)\}', content, re.DOTALL).group(0)
new_analyze = """async def _process_analyze_media(url: str, sample_interval: float, top_k: int, threshold: float) -> dict:
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
    }"""
content = content.replace(old_analyze, new_analyze)

# Update _process_detect_nsfw
old_nsfw = re.search(r'async def _process_detect_nsfw.*?return \{(.*?)\}', content, re.DOTALL).group(0)
new_nsfw = """async def _process_detect_nsfw(url: str, sample_interval: float) -> dict:
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
    }"""
content = content.replace(old_nsfw, new_nsfw)

with open('api/handlers.py', 'w') as f:
    f.write(content)
print("Updated successfully via regex")
