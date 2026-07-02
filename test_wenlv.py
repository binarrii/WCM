import asyncio
import httpx

async def test():
    b64_img = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
    
    url = "https://models.ai.wtvdev.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-o8EGlzXqMQi8Ba06E2B1BcF8217c45B6Bb70Ce5765B70c42"
    }
    payload = {
        "model": "WasuAI/Wenlv-Max",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "max_tokens": 64,
        "temperature": 0.3
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            print(resp.status_code, resp.text)
        except Exception as e:
            print(e)

asyncio.run(test())
