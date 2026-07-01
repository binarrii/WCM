import requests

try:
    resp = requests.post("http://10.252.25.251:8000/api/v1/detect_nsfw", json={"frames_data": [[0.0, "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="]]}, timeout=10)
    print(resp.json())
except Exception as e:
    print(f"Error: {e}")
