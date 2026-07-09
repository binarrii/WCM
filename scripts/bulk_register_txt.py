import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import time
import asyncio
import uuid

import cv2
import numpy as np


import httpx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from deepface import DeepFace

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wcm_facerec.database import Person
from wcm_facerec.face_engine import _persist_image


PERSONS_TXT = "/home/aigc/wcm/persons.txt"
MAX_WORKERS = 10


CATEGORY_MAP = {
    "人物": {"type": "普通人物", "remarks": ""},
    "劣迹": {"type": "劣迹艺人", "remarks": ""},
    "敏感人物": {"type": "时政敏感", "remarks": ""},
    "落马官员": {"type": "落马官员", "remarks": ""},
    "落马": {"type": "落马官员", "remarks": ""},
    "邪教人物": {"type": "邪教人物", "remarks": ""},
}


def _create_person(name: str, category: str) -> Person:
    try:
        cat_info = CATEGORY_MAP.get(category, {"type": "其他", "remarks": ""})
        person = Person(
            id=uuid.uuid4(),
            name=name,
            occupation="",
            type_=cat_info["type"],
            remarks=cat_info["remarks"],
        )
        return person
    except:
        pass

def _extract_info_from_path(img_path: str):
    """
    Extract category and name from path.
    Example: /data/wcm/落马官员/liuzhijun_82db22508712afca35f575c6ae1571a3.png
    """
    path_obj = Path(img_path)
    category = path_obj.parent.name
    filename = path_obj.stem
    # Remove hash suffix if exists
    parts = filename.rsplit('_', 1)
    if len(parts) == 2 and len(parts[1]) >= 32:  
        name = parts[0]
    else:
        name = filename
    return category, name

async def async_register(img_path, person, name, category):
    ext = Path(img_path).suffix.lower() or None
    with open(img_path, "rb") as f:
        image_bytes = f.read()

    # Resize image_bytes if long side > 1280
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > 1280:
            scale = 1280.0 / long_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            success, buf = cv2.imencode('.jpg', resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if success:
                image_bytes = buf.tobytes()

    # 拷贝到业务目录
    persisted_path = _persist_image(image_bytes, name, category, ext='.jpg')

    # 调用注册接口
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {
                "file": (os.path.basename(persisted_path), image_bytes, f"image/{ext}"),
            }
            data = {
                "name": name,
                "category": category,
                "person_id": str(person.id),
            }
            resp = await client.post(
                "http://localhost:8000/api/v1/register",
                data=data,
                files=files,
            )
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text}")

            return f"[Success] {img_path} - Registered and linked to Person({name})."
    except Exception as e:
        return f"[Error] {img_path} - API failed: {str(e)}"

def process_image(img_path):
    try:
        img_path = img_path.strip()
        if not os.path.exists(img_path):
            return f"[Failed] {img_path} - File does not exist."
            
        category, name = _extract_info_from_path(img_path)
        
        # No local face detection - let the API handle it
        person = _create_person(name, category)
        
        # 严格遵循 register_from_image 流程入库
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                async_register(img_path, person, name, category)
            )
            return result
        finally:
            loop.close()

    except Exception as e:
        return f"[Error] {img_path} - Processing failed: {str(e)}"

def main():
    if not os.path.exists(PERSONS_TXT):
        print(f"File not found: {PERSONS_TXT}")
        return
        
    with open(PERSONS_TXT, "r") as f:
        image_files = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(image_files)} image paths in {PERSONS_TXT}.")
    
    if not image_files:
        print("No images found. Exiting.")
        return
        
    print(f"Starting registration with {MAX_WORKERS} threads...")
    start_time = time.time()

    success_count = 0
    skipped_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for _img in image_files:
            if not os.path.exists(_img):
                skipped_count += 1
                print(f"Not exist: {_img}")
                continue
            try:
                with open(_img, "rb") as f:
                    image_bytes = f.read()
                    image_array = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                face_objs = DeepFace.extract_faces(
                    img, detector_backend="fastmtcnn", enforce_detection=False
                )
                if len(face_objs) > 1:
                    skipped_count += 1
                    print(f"More than one face: {_img}")
                    continue
            except:
                error_count += 1

            future_to_img = {executor.submit(process_image, _img): _img}

        for future in as_completed(future_to_img):
            result = future.result()
            print(result)
            
            if result.startswith("[Success]"):
                success_count += 1
            elif result.startswith("[Skipped]"):
                skipped_count += 1
            else:
                error_count += 1
                
    elapsed = time.time() - start_time
    print("-" * 40)
    print("Batch Registration Completed!")
    print(f"Time Elapsed: {elapsed:.2f} seconds")
    print(f"Total Processed: {len(image_files)}")
    print(f"  - Success: {success_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Failed/Error: {error_count}")

if __name__ == "__main__":
    main()
