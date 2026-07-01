import re

with open('api/routes.py', 'r') as f:
    content = f.read()

utils_functions = ['run_in_inference_thread', '_download_url_safe', '_download_video_safe_sync', '_extract_video_frames_for_ocr']
handlers_functions = ['_verify_candidates', '_crop_largest_face', '_detect_and_crop_face', 
                      '_detect_and_crop_face_from_bytes', '_search_video_frames', '_search_face_in_image', 
                      '_call_ocr_api', '_call_llm_guard', '_process_detect_sensitive', 'get_nsfw_pipeline', 
                      'detect_visual_nsfw', '_call_qwen_nsfw', '_process_analyze_media']

# We can just parse the file using ast and extract the source code of each function, but we also need comments and decorators.
# It's better to just read it block by block or do it with string matching.
