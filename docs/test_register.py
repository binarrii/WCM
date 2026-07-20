import httpx
import base64
import sys

api_url = "http://10.252.25.251:5000"
img_path = "/Users/binarii/Downloads/2026智能审核/劣迹艺人/0000000000016508_164ad27ae0448e990c63dc3e9a2474cc.jpg"

try:
    with open(img_path, "rb") as f:
        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

payload = {
    "img": img_b64,
    "img_name": "test_uuid",
    "model_name": "Facenet512",
    "detector_backend": "fastmtcnn",
    "align": True,
    "enforce_detection": False,
}

print("Sending request to /register...")
resp = httpx.post(f"{api_url}/register", json=payload, timeout=60.0)
print(f"Status Code: {resp.status_code}")
print("Response body:")
print(resp.text)
