import os

utils_imports = """import asyncio
import httpx
import cv2
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from wcm_facerec.config import settings
from wcm_facerec import __version__

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
MIN_FACE_PIXELS = 64 * 64

# Dedicated single-thread pool for CUDA/DeepFace inference
inference_executor = ThreadPoolExecutor(max_workers=1)

"""

handlers_imports = """import asyncio
import base64
import json
import os
import uuid
import io
from pathlib import Path
from typing import Union
import cv2
import httpx
import numpy as np
import torch
from transformers import pipeline
from PIL import Image

from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine
from .utils import (
    run_in_inference_thread, 
    _download_url_safe, 
    _download_video_safe_sync, 
    _extract_video_frames_for_ocr,
    VIDEO_EXTENSIONS,
    MIN_FACE_PIXELS
)

nsfw_pipeline = None

"""

routes_imports = """\"\"\"API routes for face recognition service.\"\"\"
import uuid
import json
from pathlib import Path
from typing import Union
import asyncio
import base64
import cv2
import numpy as np

from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine
from .utils import (
    run_in_inference_thread,
    _download_url_safe,
    VIDEO_EXTENSIONS
)
from .handlers import (
    _verify_candidates,
    _detect_and_crop_face_from_bytes,
    _search_video_frames,
    _process_detect_sensitive,
    _process_analyze_media
)

api_bp = APIRouter()

"""

def process_file(src, dest, imports_text, remove_lines=0):
    with open(src, 'r') as f:
        content = f.read()
    
    # Optionally strip top boilerplate from the original parsed file
    if src == 'api/routes_new.py':
        # Remove everything before @api_bp.get("/health")
        idx = content.find('@api_bp.get("/health")')
        if idx != -1:
            content = content[idx:]

    with open(dest, 'w') as f:
        f.write(imports_text + content)


def main():
    process_file('api/utils_new.py', 'api/utils.py', utils_imports)
    process_file('api/handlers_new.py', 'api/handlers.py', handlers_imports)
    process_file('api/routes_new.py', 'api/routes.py', routes_imports)

    # Cleanup temp files
    os.remove('api/utils_new.py')
    os.remove('api/handlers_new.py')
    os.remove('api/routes_new.py')

if __name__ == "__main__":
    main()
