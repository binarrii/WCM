import requests, base64, cv2, numpy as np
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.imwrite("dummy.jpg", img)
with open("dummy.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
resp = requests.post("http://10.252.25.251:5000/search", json={
    "img": f"data:image/jpeg;base64,{b64}",
    "model_name": "Facenet512",
    "detector_backend": "retinaface"
})
print(resp.status_code, resp.text)
