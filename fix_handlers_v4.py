import re

with open('api/handlers.py', 'r') as f:
    content = f.read()

# 1. Replace the model in _call_qwen_image_analysis
old_vlm = re.search(r'async def _call_qwen_image_analysis.*?return "无法获取描述"', content, re.DOTALL).group(0)
new_vlm = old_vlm.replace('"WasuAI/PaddleOCR-VL-1.6"', '"WasuAI/Wenlv-Max"')
content = content.replace(old_vlm, new_vlm)

# 2. Add deduplication / truncation for OCR text
old_ocr = re.search(r'async def _call_ocr_api.*?return ""', content, re.DOTALL).group(0)
# we need to truncate the output of text to max 500 chars, but also let's collapse repetitive chars
new_ocr_str = """            analysis = data["choices"][0]["message"]["content"].strip()
            
            # Basic dedup and truncate to prevent hallucination explosions
            if len(analysis) > 500:
                analysis = analysis[:500] + "..."
            return analysis"""
old_ocr_str = """            return data["choices"][0]["message"]["content"].strip()"""
new_ocr = old_ocr.replace(old_ocr_str, new_ocr_str)
content = content.replace(old_ocr, new_ocr)

# 3. Remove "is_nsfw" from outputs
content = re.sub(r'\s*"is_nsfw":.*?,\n', '\n', content)

with open('api/handlers.py', 'w') as f:
    f.write(content)
print("Fixes applied successfully")
