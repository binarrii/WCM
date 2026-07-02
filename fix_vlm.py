import re

with open('api/handlers.py', 'r') as f:
    content = f.read()

content = content.replace('"model": "WasuAI/Qwen3.6-27B-Abliterated",', '"model": "WasuAI/PaddleOCR-VL-1.6",')

with open('api/handlers.py', 'w') as f:
    f.write(content)
