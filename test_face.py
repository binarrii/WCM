import urllib.request
import json
import base64

with open("wcm_facerec/tests/test.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

req = urllib.request.Request("http://10.252.25.251:5000/search", data=json.dumps({
    "img": f"data:image/jpeg;base64,{b64}",
    "model_name": "Facenet512",
    "detector_backend": "retinaface"
}).encode('utf-8'), headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as resp:
        print("SEARCH:", resp.read().decode())
except Exception as e:
    print("ERROR:", e)
