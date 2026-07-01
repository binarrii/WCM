import faulthandler
faulthandler.enable()
print("1")
from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect
print("2")
from wcm_facerec import __version__
print("3")
from wcm_facerec.config import settings
print("4")
from wcm_facerec.face_engine import get_face_engine
print("5")
from api.utils import run_in_inference_thread, _download_url_safe, VIDEO_EXTENSIONS
print("6")
from api.handlers import _verify_candidates, _detect_and_crop_face_from_bytes, _detect_and_crop_face, _search_video_frames, _process_detect_sensitive, _process_detect_nsfw, _process_analyze_media
print("7")
