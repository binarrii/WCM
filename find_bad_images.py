import os
import glob
import cv2
import numpy as np
from deepface import DeepFace

folders = [
    "/Users/binarii/Downloads/2026智能审核/明星艺人",
    "/Users/binarii/Downloads/2026智能审核/人物",
    "/Users/binarii/Downloads/2026智能审核/时政敏感"
]

images = []
for f in folders:
    images.extend(glob.glob(os.path.join(f, "*.png")))
    images.extend(glob.glob(os.path.join(f, "*.jpg")))

found = 0
for img_path in images:
    if found >= 3:
        break
    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        faces = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=False)
        
        # for opencv, require at least 2 faces with confidence > 0.9
        valid_faces = [f for f in faces if f['confidence'] > 0.9]
        
        if len(valid_faces) >= 2:
            print(f"SUCCESS: Found {len(valid_faces)} faces in: {img_path}")
            found += 1
            
    except Exception as e:
        pass
