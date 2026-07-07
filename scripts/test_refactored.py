import asyncio
from pathlib import Path
from wcm_facerec.face_engine import get_face_engine

async def main():
    engine = get_face_engine()
    print("Testing FaceEngine (API Delegated)")
    
    # URL of a sample face image
    test_img = "https://avatars.githubusercontent.com/u/1?v=4"
    
    print(f"\n--- Testing Register ---")
    try:
        record = await engine.register_from_image(
            name="Test User",
            img_source=test_img
        )
        print(f"Successfully registered. Person ID: {record.id}")
    except Exception as e:
        print(f"Register failed: {e}")
        return
        
    print(f"\n--- Testing Search ---")
    try:
        results = await engine.search(
            img_source=test_img,
            top_k=5,
            threshold=0.6
        )
        print(f"Search found {len(results)} matches.")
        for i, r in enumerate(results):
            print(f"  Match {i+1}: Name={r.get('person', {}).get('name')}, Distance={r.get('distance')}, X={r.get('source_x')}")
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
