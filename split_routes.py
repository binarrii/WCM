import re
from pathlib import Path

def main():
    with open('api/routes.py', 'r') as f:
        lines = f.readlines()

    utils_funcs = ['run_in_inference_thread', '_download_url_safe', '_download_video_safe_sync', '_extract_video_frames_for_ocr']
    handler_funcs = ['_verify_candidates', '_crop_largest_face', '_detect_and_crop_face', 
                     '_detect_and_crop_face_from_bytes', '_search_video_frames', '_search_face_in_image',
                     '_call_ocr_api', '_call_llm_guard', '_process_detect_sensitive', 'get_nsfw_pipeline',
                     'detect_visual_nsfw', '_call_qwen_nsfw', '_process_analyze_media']

    def find_bounds(func_name):
        start_idx = -1
        # Find def or async def
        for i, line in enumerate(lines):
            if line.startswith(f"def {func_name}(") or line.startswith(f"async def {func_name}(") or line.startswith(f"def {func_name}:") or line.startswith(f"async def {func_name}:"):
                start_idx = i
                break
        if start_idx == -1: return -1, -1
        
        # Check decorators above
        while start_idx > 0 and lines[start_idx-1].startswith('@'):
            start_idx -= 1

        # Find end_idx (next function or end of file)
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith("def ") or lines[i].startswith("async def ") or lines[i].startswith("class ") or lines[i].startswith("@"):
                end_idx = i
                break
        
        # Retreat blank lines
        while end_idx > start_idx and lines[end_idx-1].strip() == "":
            end_idx -= 1
            
        return start_idx, end_idx

    utils_code = []
    handlers_code = []
    
    # We will replace the original functions with empty string in routes_code, 
    # but let's just build routes_code by omitting them.
    routes_code = []
    
    skip_until = -1
    for i, line in enumerate(lines):
        if i < skip_until:
            continue
            
        is_matched = False
        for func in utils_funcs:
            s, e = find_bounds(func)
            if s == i:
                utils_code.append("".join(lines[s:e]) + "\n\n")
                skip_until = e
                is_matched = True
                break
        if is_matched: continue
        
        for func in handler_funcs:
            s, e = find_bounds(func)
            if s == i:
                handlers_code.append("".join(lines[s:e]) + "\n\n")
                skip_until = e
                is_matched = True
                break
        if is_matched: continue
        
        routes_code.append(line)

    # Note: ns_pipeline = None is a global var for nsfw
    # I should manually move it to handlers.
    
    with open('api/utils_new.py', 'w') as f:
        f.write("".join(utils_code))
        
    with open('api/handlers_new.py', 'w') as f:
        f.write("".join(handlers_code))
        
    with open('api/routes_new.py', 'w') as f:
        f.write("".join(routes_code))

if __name__ == "__main__":
    main()
