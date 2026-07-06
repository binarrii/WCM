import asyncio
import httpx

async def test_deepface_api():
    async with httpx.AsyncClient() as client:
        # test register
        print("Testing register...")
        with open("/Users/binarii/workspaces/binarii/WCM/test.jpg", "wb") as f:
            f.write(b"fake image data")
            
test_deepface_api()
