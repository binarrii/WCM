import os
import re
import subprocess
import urllib.parse
import base64
import mimetypes

html_content = subprocess.check_output(['npx', 'marked', 'dataset_creation_standards.md'], text=True)

def replace_image_with_base64(match):
    file_path = match.group(1)
    file_path = urllib.parse.unquote(file_path)
    try:
        with open(file_path, "rb") as img_file:
            b64_data = base64.b64encode(img_file.read()).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "image/png"
            return f'src="data:{mime_type};base64,{b64_data}"'
    except Exception as e:
        print(f"Warning: Could not load image {file_path}: {e}")
        return match.group(0)

html_content = re.sub(r'src="file://([^"]+)"', replace_image_with_base64, html_content)

# Generate TOC
toc = "<h2>目录 / 大纲</h2>\n<ul style='list-style-type: none; padding-left: 0;'>\n"
for match in re.finditer(r'<h([23])>(.*?)</h\1>', html_content):
    level = int(match.group(1))
    text = match.group(2)
    text_clean = re.sub(r'<[^>]+>', '', text)
    indent = "20px" if level == 3 else "0px"
    weight = "bold" if level == 2 else "normal"
    toc += f"<li style='margin-left: {indent}; margin-bottom: 5px;'><a href='#{text_clean}' style='text-decoration: none; color: #2563eb; font-weight: {weight};'>{text_clean}</a></li>\n"
toc += "</ul>\n<hr>\n"

# Add id attributes for anchors
html_content = re.sub(r'<h([23])>(.*?)</h\1>', lambda m: f'<h{m.group(1)} id="{re.sub(r"<[^>]+>", "", m.group(2))}">{m.group(2)}</h{m.group(1)}>', html_content)

# Watermark SVG
svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="500" height="500">
<g transform="rotate(-35, 250, 250)">
<text x="50%" y="50%" fill="rgba(0,0,0,0.06)" font-size="26" font-weight="bold" font-family="sans-serif" text-anchor="middle">平台技术中心      AI能力研发分部</text>
</g>
</svg>'''
svg_encoded = urllib.parse.quote(svg)

watermark_css = f"""
body::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 9999;
    background-image: url('data:image/svg+xml;utf8,{svg_encoded}');
    background-repeat: repeat;
}}
"""

template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多模态审核模型：旗帜、地图与敏感人物数据集制作标准</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #374151;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #f3f4f6;
        }}
        {watermark_css}
        h1, h2, h3 {{ color: #111827; }}
        h1 {{ font-size: 2.2rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; margin-bottom: 30px; }}
        h2 {{ font-size: 1.5rem; margin-top: 40px; color: #2563eb; border-left: 4px solid #2563eb; padding-left: 12px; }}
        h3 {{ font-size: 1.25rem; margin-top: 25px; color: #4b5563; }}
        ul, ol {{ margin-bottom: 20px; }}
        li {{ margin-bottom: 8px; }}
        p {{ margin-bottom: 16px; }}
        strong {{ color: #111827; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin: 10px 0 20px 0; display: block; }}
        hr {{ border: 0; height: 1px; background: #e5e7eb; margin: 40px 0; }}
        .container {{ background-color: #ffffff; padding: 50px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); position: relative; z-index: 1; }}
    </style>
</head>
<body>
<div class="container">
{toc}
{html_content}
</div>
</body>
</html>"""

with open("dataset_creation_standards.html", "w") as f:
    f.write(template)
print("Done!")
