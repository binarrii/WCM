import re

with open('api/handlers.py', 'r') as f:
    content = f.read()

# Update _call_llm_guard
old_llm_guard = """async def _call_llm_guard(text: str) -> dict:
    if not text.strip():
        return {"safe": True, "reason": "No text extracted from the media."}
    
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/WasuGuard-Gen-4B",
        "messages": [
            {
                "role": "system",
                "content": "You are a sensitive information guard. Analyze the text for sensitive, prohibited, or inappropriate information. Output your analysis clearly."
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
            return {"analysis": data["choices"][0]["message"]["content"]}
        except Exception as e:
            if hasattr(e, 'response') and getattr(e, 'response') is not None:
                print(f"LLM Guard Error: {e.response.text}")
            else:
                print(f"LLM Guard Error: {e}")
            return {"error": str(e), "status": "Model API failed or unreachable"}"""

new_llm_guard = """async def _call_llm_guard(text: str) -> dict:
    if not text.strip():
        return {"safe": True, "analysis": "No text extracted from the media."}
    
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/WasuGuard-Gen-4B",
        "messages": [
            {
                "role": "system",
                "content": "You are a sensitive information guard. Analyze the text for sensitive, prohibited, or inappropriate information. If the text is completely safe and contains no sensitive/prohibited information, you MUST output exactly 'SAFE'. If it contains sensitive information, output 'UNSAFE:' followed by a clear description of what is sensitive."
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
            is_safe = analysis.upper().startswith("SAFE")
            return {"safe": is_safe, "analysis": analysis}
        except Exception as e:
            if hasattr(e, 'response') and getattr(e, 'response') is not None:
                print(f"LLM Guard Error: {e.response.text}")
            else:
                print(f"LLM Guard Error: {e}")
            return {"safe": True, "error": str(e), "status": "Model API failed or unreachable"}"""

# Update _call_qwen_nsfw
old_qwen_nsfw = """async def _call_qwen_nsfw(text: str) -> dict:
    if not text.strip():
        return {"safe": True, "reason": "No text extracted from the media."}
    
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/Qwen3.6-27B-Abliterated",
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the following text for sensitive, NSFW, prohibited, or inappropriate information. Output your analysis clearly. Text: {text}"
            }
        ],
        "temperature": 0.6,
        "reasoning_effort": "none"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return {"analysis": data["choices"][0]["message"]["content"]}
        except Exception as e:
            if hasattr(e, 'response') and getattr(e, 'response') is not None:
                print(f"Qwen API Error: {e.response.text}")
            else:
                print(f"Qwen API Error: {e}")
            return {"error": str(e), "status": "Model API failed or unreachable"}"""

new_qwen_nsfw = """async def _call_qwen_nsfw(text: str) -> dict:
    if not text.strip():
        return {"safe": True, "analysis": "No text extracted from the media."}
    
    url = settings.model_api_url
    headers = {
        "Authorization": f"Bearer {settings.model_api_key}"
    }
    payload = {
        "model": "WasuAI/Qwen3.6-27B-Abliterated",
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the following text for sensitive, NSFW, prohibited, or inappropriate information. If the text is completely safe and contains no such information, you MUST output exactly 'SAFE'. If it is unsafe, output 'UNSAFE:' followed by a clear description of the violation.\n\nText: {text}"
            }
        ],
        "temperature": 0.6,
        "reasoning_effort": "none"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            analysis = data["choices"][0]["message"]["content"].strip()
            is_safe = analysis.upper().startswith("SAFE")
            return {"safe": is_safe, "analysis": analysis}
        except Exception as e:
            if hasattr(e, 'response') and getattr(e, 'response') is not None:
                print(f"Qwen API Error: {e.response.text}")
            else:
                print(f"Qwen API Error: {e}")
            return {"safe": True, "error": str(e), "status": "Model API failed or unreachable"}"""

# Replace _process_detect_sensitive
old_process_sensitive = """async def _process_detect_sensitive(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    extracted_texts = []
    
    if is_video:
        video_path = Path(f"/tmp/guard_video_{os.urandom(8).hex()}.mp4")
        try:
            await asyncio.to_thread(
                _download_video_safe_sync, url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024
            )
            frames_data = await asyncio.to_thread(_extract_video_frames_for_ocr, video_path, sample_interval)
            
            for timestamp, b64_img in frames_data:
                text = await _call_ocr_api(b64_img)
                if text:
                    extracted_texts.append(f"[Frame {timestamp:.1f}s]:\n{text}")
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        text = await _call_ocr_api(b64_img)
        if text:
            extracted_texts.append(text)

    full_text = "\n\n".join(extracted_texts)
    guard_result = await _call_llm_guard(full_text)
    
    return {
        "extracted_text": full_text,
        "guard_result": guard_result
    }"""

new_process_sensitive = """async def _process_detect_sensitive(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    unsafe_text_frames = []
    
    async def _analyze_text(timestamp, b64_img):
        text = await _call_ocr_api(b64_img)
        if text:
            guard = await _call_llm_guard(text)
            if not guard.get("safe", True):
                return {"timestamp": timestamp, "description": guard.get("analysis", "")}
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

# Replace _process_analyze_media
old_process_analyze = """async def _process_analyze_media(url: str, sample_interval: float, top_k: int, threshold: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    engine = get_face_engine()
    
    extracted_texts = []
    nsfw_visual_results = []
    face_search_results = []
    
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
            
            for timestamp, b64_img in frames_data:
                img_bytes = base64.b64decode(b64_img)
                visual_result = await run_in_inference_thread(detect_visual_nsfw, img_bytes)
                nsfw_visual_results.append({
                    "timestamp": timestamp,
                    "result": visual_result
                })
                
                text = await _call_ocr_api(b64_img)
                if text:
                    extracted_texts.append(f"[Frame {timestamp:.1f}s]:\n{text}")
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

        visual_result = await run_in_inference_thread(detect_visual_nsfw, img_bytes)
        nsfw_visual_results.append({
            "type": "image",
            "result": visual_result
        })
        
        text = await _call_ocr_api(b64_img)
        if text:
            extracted_texts.append(text)

    full_text = "\n\n".join(extracted_texts)
    text_guard_result = await _call_llm_guard(full_text)
    
    is_nsfw = False
    high_confidence_nsfw_frames = []
    for res in nsfw_visual_results:
        for label_score in res.get("result", []):
            if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                is_nsfw = True
                frame_info = {"confidence": label_score["score"]}
                if "timestamp" in res:
                    frame_info["timestamp"] = res["timestamp"]
                else:
                    frame_info["type"] = "image"
                high_confidence_nsfw_frames.append(frame_info)
                break
                
    return {
        "face_search_results": face_search_results,
        "is_nsfw": is_nsfw,
        "visual_analysis": high_confidence_nsfw_frames,
        "extracted_text": full_text,
        "text_analysis": text_guard_result
    }"""

new_process_analyze = """async def _process_analyze_media(url: str, sample_interval: float, top_k: int, threshold: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    engine = get_face_engine()
    
    unsafe_text_frames = []
    nsfw_visual_results = []
    face_search_results = []
    
    async def _analyze_frame(timestamp, b64_img):
        img_bytes = base64.b64decode(b64_img)
        visual_result = await run_in_inference_thread(detect_visual_nsfw, img_bytes)
        
        text = await _call_ocr_api(b64_img)
        text_guard = None
        if text:
            text_guard = await _call_llm_guard(text)
            
        return timestamp, visual_result, text_guard

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
            
            for timestamp, visual_result, text_guard in frame_results:
                nsfw_visual_results.append({
                    "timestamp": timestamp,
                    "result": visual_result
                })
                if text_guard and not text_guard.get("safe", True):
                    unsafe_text_frames.append({
                        "timestamp": timestamp,
                        "description": text_guard.get("analysis", "")
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

        _, visual_result, text_guard = await _analyze_frame(None, b64_img)
        nsfw_visual_results.append({
            "type": "image",
            "result": visual_result
        })
        if text_guard and not text_guard.get("safe", True):
            unsafe_text_frames.append({
                "type": "image",
                "description": text_guard.get("analysis", "")
            })

    is_nsfw = False
    high_confidence_nsfw_frames = []
    for res in nsfw_visual_results:
        for label_score in res.get("result", []):
            if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                is_nsfw = True
                frame_info = {"confidence": label_score["score"]}
                if "timestamp" in res:
                    frame_info["timestamp"] = res["timestamp"]
                else:
                    frame_info["type"] = "image"
                high_confidence_nsfw_frames.append(frame_info)
                break
                
    return {
        "face_search_results": face_search_results,
        "is_nsfw": is_nsfw,
        "visual_analysis": high_confidence_nsfw_frames,
        "unsafe_text_frames": unsafe_text_frames
    }"""

# Replace _process_detect_nsfw
old_process_nsfw = """async def _process_detect_nsfw(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    
    extracted_texts = []
    nsfw_visual_results = []
    
    if is_video:
        video_path = Path(f"/tmp/nsfw_video_{os.urandom(8).hex()}.mp4")
        try:
            await asyncio.to_thread(
                _download_video_safe_sync, url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024
            )
            frames_data = await asyncio.to_thread(_extract_video_frames_for_ocr, video_path, sample_interval)
            
            for timestamp, b64_img in frames_data:
                img_bytes = base64.b64decode(b64_img)
                visual_result = await run_in_inference_thread(detect_visual_nsfw, img_bytes)
                nsfw_visual_results.append({
                    "timestamp": timestamp,
                    "result": visual_result
                })
                
                text = await _call_ocr_api(b64_img)
                if text:
                    extracted_texts.append(f"[Frame {timestamp:.1f}s]:\n{text}")
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        visual_result = await run_in_inference_thread(detect_visual_nsfw, img_bytes)
        nsfw_visual_results.append({
            "type": "image",
            "result": visual_result
        })
        
        text = await _call_ocr_api(b64_img)
        if text:
            extracted_texts.append(text)

    full_text = "\n\n".join(extracted_texts)
    text_guard_result = await _call_qwen_nsfw(full_text)
    
    is_nsfw = False
    high_confidence_nsfw_frames = []
    for res in nsfw_visual_results:
        for label_score in res.get("result", []):
            if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                is_nsfw = True
                frame_info = {"confidence": label_score["score"]}
                if "timestamp" in res:
                    frame_info["timestamp"] = res["timestamp"]
                else:
                    frame_info["type"] = "image"
                high_confidence_nsfw_frames.append(frame_info)
                break
                
    return {
        "is_nsfw": is_nsfw,
        "visual_analysis": high_confidence_nsfw_frames,
        "extracted_text": full_text,
        "text_analysis": text_guard_result
    }"""

new_process_nsfw = """async def _process_detect_nsfw(url: str, sample_interval: float) -> dict:
    is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
    
    unsafe_text_frames = []
    nsfw_visual_results = []
    
    async def _analyze_frame(timestamp, b64_img):
        img_bytes = base64.b64decode(b64_img)
        visual_result = await run_in_inference_thread(detect_visual_nsfw, img_bytes)
        
        text = await _call_ocr_api(b64_img)
        text_guard = None
        if text:
            text_guard = await _call_qwen_nsfw(text)
            
        return timestamp, visual_result, text_guard
        
    if is_video:
        video_path = Path(f"/tmp/nsfw_video_{os.urandom(8).hex()}.mp4")
        try:
            await asyncio.to_thread(
                _download_video_safe_sync, url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024
            )
            frames_data = await asyncio.to_thread(_extract_video_frames_for_ocr, video_path, sample_interval)
            
            tasks = [_analyze_frame(ts, b64) for ts, b64 in frames_data]
            frame_results = await asyncio.gather(*tasks)
            
            for timestamp, visual_result, text_guard in frame_results:
                nsfw_visual_results.append({
                    "timestamp": timestamp,
                    "result": visual_result
                })
                if text_guard and not text_guard.get("safe", True):
                    unsafe_text_frames.append({
                        "timestamp": timestamp,
                        "description": text_guard.get("analysis", "")
                    })
        finally:
            if video_path.exists():
                video_path.unlink()
    else:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        _, visual_result, text_guard = await _analyze_frame(None, b64_img)
        nsfw_visual_results.append({
            "type": "image",
            "result": visual_result
        })
        if text_guard and not text_guard.get("safe", True):
            unsafe_text_frames.append({
                "type": "image",
                "description": text_guard.get("analysis", "")
            })

    is_nsfw = False
    high_confidence_nsfw_frames = []
    for res in nsfw_visual_results:
        for label_score in res.get("result", []):
            if label_score["label"] == "nsfw" and label_score["score"] > 0.5:
                is_nsfw = True
                frame_info = {"confidence": label_score["score"]}
                if "timestamp" in res:
                    frame_info["timestamp"] = res["timestamp"]
                else:
                    frame_info["type"] = "image"
                high_confidence_nsfw_frames.append(frame_info)
                break
                
    return {
        "is_nsfw": is_nsfw,
        "visual_analysis": high_confidence_nsfw_frames,
        "unsafe_text_frames": unsafe_text_frames
    }"""

content = content.replace(old_llm_guard, new_llm_guard)
content = content.replace(old_qwen_nsfw, new_qwen_nsfw)
content = content.replace(old_process_sensitive, new_process_sensitive)
content = content.replace(old_process_analyze, new_process_analyze)
content = content.replace(old_process_nsfw, new_process_nsfw)

with open('api/handlers.py', 'w') as f:
    f.write(content)
print("Updated successfully")
