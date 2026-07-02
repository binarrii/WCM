import re

with open('api/handlers.py', 'r') as f:
    content = f.read()

old_ocr_str = """            import re
            # Remove massive consecutive repetition (hallucinations like 王晓燕王晓燕...)"""
            
new_ocr_str = """            import re
            # Clean up <|LOC_X|> bounding box tokens that the VLM might output
            analysis = re.sub(r'<\\|LOC_\\d+\\|>', '', analysis)
            
            # Remove massive consecutive repetition (hallucinations like 王晓燕王晓燕...)"""

content = content.replace(old_ocr_str, new_ocr_str)

with open('api/handlers.py', 'w') as f:
    f.write(content)
