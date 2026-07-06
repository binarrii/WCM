import requests

def test():
    print("Testing DeepFace /search API...")
    url = "http://10.252.25.251:5000/search"
    img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    
    payload = {
        "img": "data:image/png;base64," + img,
        "model_name": "Facenet512",
        "distance_metric": "cosine",
        "enforce_detection": False
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        print("Status:", r.status_code)
        print("Response:", r.text)
    except Exception as e:
        print("Error:", e)

test()
