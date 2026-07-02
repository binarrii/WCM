import re

with open('api/handlers.py', 'r') as f:
    content = f.read()

old_ocr_str = """            # Basic dedup and truncate to prevent hallucination explosions
            if len(analysis) > 500:
                analysis = analysis[:500] + "..."
            return analysis"""
            
new_ocr_str = """            import re
            # Remove massive consecutive repetition (hallucinations like 王晓燕王晓燕...)
            analysis = re.sub(r'(.{1,30}?)\\1{4,}', r'\\1...', analysis)
            if len(analysis) > 500:
                analysis = analysis[:500] + "..."
            return analysis"""

content = content.replace(old_ocr_str, new_ocr_str)

with open('api/handlers.py', 'w') as f:
    f.write(content)
