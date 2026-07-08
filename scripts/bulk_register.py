import os
import glob
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from deepface import DeepFace

# ==========================================
# 配置区域
# ==========================================
IMAGE_DIR = "/data/wcm"
# 注：原生的 deepface-api 并没有 /register 接口，这里默认指向 WCM 的业务系统注册接口。
# 如果您在其他地方实现了特定的 deepface-api 注册接口，请修改下方 URL。
REGISTER_API_URL = "http://127.0.0.1:8000/api/v1/register"
MAX_WORKERS = 10
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
# ==========================================

def get_image_files(directory):
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext.upper()}"), recursive=True))
    return files

def process_image(img_path):
    filename = os.path.basename(img_path)
    
    # 1. 使用 DeepFace 本地检测人脸数量
    try:
        # 使用 retinaface 会更准确，如果为了速度可以改成 'opencv' 或 'mtcnn'
        # enforce_detection=True 表示如果没检测到人脸会抛出 ValueError
        faces = DeepFace.extract_faces(
            img_path=img_path, 
            detector_backend="retinaface", 
            enforce_detection=True
        )
    except ValueError:
        return f"[Skipped] {filename} - No face detected."
    except Exception as e:
        return f"[Error] {filename} - Local detection failed: {str(e)}"
        
    face_count = len(faces)
    if face_count > 2:
        return f"[Skipped] {filename} - Found {face_count} faces (more than 2)."
        
    # 2. 调用 /register 接口进行注册
    try:
        with open(img_path, "rb") as f:
            files = {
                "file": (filename, f, "image/jpeg" if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg') else "image/png")
            }
            data = {
                "name": os.path.splitext(filename)[0],  # 使用去掉后缀的文件名作为注册名称
                "category": "batch_import"
            }
            
            resp = requests.post(REGISTER_API_URL, files=files, data=data, timeout=60)
            
            if resp.status_code == 200:
                return f"[Success] {filename} - Registered successfully."
            else:
                return f"[Failed] {filename} - API returned {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"[Error] {filename} - API Request failed: {str(e)}"

def main():
    print(f"Scanning directory: {IMAGE_DIR}")
    image_files = get_image_files(IMAGE_DIR)
    print(f"Found {len(image_files)} image files.")
    
    if not image_files:
        print("No images found. Exiting.")
        return
        
    print(f"Starting registration with {MAX_WORKERS} threads...")
    start_time = time.time()
    
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_img = {executor.submit(process_image, img_path): img_path for img_path in image_files}
        
        # 收集结果
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
