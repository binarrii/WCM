import re

with open('api/handlers.py', 'r') as f:
    content = f.read()

# Fix _call_llm_guard to better parse Category
old_guard = re.search(r'async def _call_llm_guard.*?return \{"safe": True, "category": ""\}', content, re.DOTALL).group(0)

new_guard = """async def _call_llm_guard(text: str) -> dict:
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
                if "safety:" in line.lower() and "unsafe" in line.lower():
                    is_safe = False
                elif "categories:" in line.lower():
                    idx = line.lower().find("categories:")
                    category = line[idx + len("categories:"):].strip()
                    
            if not is_safe and not category:
                if "unsafe" in analysis.lower():
                    # fallback extraction
                    category = "未知敏感内容"
                    
            return {"safe": is_safe, "category": category}
        except Exception as e:
            return {"safe": True, "category": ""}"""
content = content.replace(old_guard, new_guard)

# Add 'text' to unsafe_text_frames in _process_detect_sensitive
old_sens = re.search(r'async def _process_detect_sensitive.*?return \{(.*?)\}', content, re.DOTALL).group(0)
new_sens = old_sens.replace(
    'return {"timestamp": timestamp, "category": guard.get("category", "")}',
    'return {"timestamp": timestamp, "category": guard.get("category", ""), "text": text}'
).replace(
    'res["type"] = "image"\n            res.pop("timestamp", None)',
    'res["type"] = "image"\n            res.pop("timestamp", None)\n            # keep text inside res'
)
content = content.replace(old_sens, new_sens)

# Add 'text' to unsafe_text_frames in _process_analyze_media
old_ana = re.search(r'async def _process_analyze_media.*?return \{(.*?)\}', content, re.DOTALL).group(0)
new_ana = old_ana.replace(
    'return timestamp, visual_result, visual_desc, text_guard',
    'return timestamp, visual_result, visual_desc, text_guard, text'
).replace(
    'for timestamp, visual_result, visual_desc, text_guard in frame_results:',
    'for timestamp, visual_result, visual_desc, text_guard, frame_text in frame_results:'
).replace(
    'unsafe_text_frames.append({\n                        "timestamp": timestamp,\n                        "category": text_guard.get("category", "")\n                    })',
    'unsafe_text_frames.append({\n                        "timestamp": timestamp,\n                        "category": text_guard.get("category", ""),\n                        "text": frame_text\n                    })'
).replace(
    '_, visual_result, visual_desc, text_guard = await _analyze_frame(None, b64_img)',
    '_, visual_result, visual_desc, text_guard, frame_text = await _analyze_frame(None, b64_img)'
).replace(
    'unsafe_text_frames.append({\n                "type": "image",\n                "category": text_guard.get("category", "")\n            })',
    'unsafe_text_frames.append({\n                "type": "image",\n                "category": text_guard.get("category", ""),\n                "text": frame_text\n            })'
)
content = content.replace(old_ana, new_ana)

# Add 'text' to unsafe_text_frames in _process_detect_nsfw
old_nsfw = re.search(r'async def _process_detect_nsfw.*?return \{(.*?)\}', content, re.DOTALL).group(0)
new_nsfw = old_nsfw.replace(
    'return timestamp, visual_result, visual_desc, text_guard',
    'return timestamp, visual_result, visual_desc, text_guard, text'
).replace(
    'for timestamp, visual_result, visual_desc, text_guard in frame_results:',
    'for timestamp, visual_result, visual_desc, text_guard, frame_text in frame_results:'
).replace(
    'unsafe_text_frames.append({\n                        "timestamp": timestamp,\n                        "category": text_guard.get("category", "")\n                    })',
    'unsafe_text_frames.append({\n                        "timestamp": timestamp,\n                        "category": text_guard.get("category", ""),\n                        "text": frame_text\n                    })'
).replace(
    '_, visual_result, visual_desc, text_guard = await _analyze_frame(None, b64_img)',
    '_, visual_result, visual_desc, text_guard, frame_text = await _analyze_frame(None, b64_img)'
).replace(
    'unsafe_text_frames.append({\n                "type": "image",\n                "category": text_guard.get("category", "")\n            })',
    'unsafe_text_frames.append({\n                "type": "image",\n                "category": text_guard.get("category", ""),\n                "text": frame_text\n            })'
)
content = content.replace(old_nsfw, new_nsfw)

with open('api/handlers.py', 'w') as f:
    f.write(content)
print("Updated successfully")
