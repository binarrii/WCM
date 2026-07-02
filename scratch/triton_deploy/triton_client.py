import httpx
import base64
import numpy as np

async def _call_triton_face_engine(engine, frame, top_k, threshold, current_frame_time):
    url = f"{settings.triton_server_url}/v2/models/face_engine/infer"
    
    # encode to jpeg then base64
    _, buffer = cv2.imencode('.jpg', frame)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    
    payload = {
        "inputs": [
            {
                "name": "IMAGE_BYTES",
                "shape": [1],
                "datatype": "BYTES",
                "data": [b64_str]
            }
        ]
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            # extract outputs
            outputs = {out["name"]: out for out in data["outputs"]}
            num_faces = outputs["NUM_FACES"]["data"][0]
            
            if num_faces == 0:
                return []
                
            bboxes = outputs["BBOXES"]["data"]
            embeddings = outputs["EMBEDDINGS"]["data"]
            
            # Flatten lists to numpy arrays
            # Triton JSON API returns flat lists for data
            bboxes = np.array(bboxes).reshape((3, 4)) # MAX_FACES = 3
            embeddings = np.array(embeddings).reshape((3, 512))
            
            all_results = []
            for i in range(num_faces):
                emb = embeddings[i]
                # we don't have face_img, but engine.search only needs embedding!
                search_res = engine.search(
                    embedding=emb,
                    name=None,
                    top_k=top_k,
                    threshold=threshold
                )
                
                # Deduplicate and append
                seen = set()
                for r in search_res:
                    r["frame_time"] = current_frame_time
                    key = (r.get("name"), r.get("person_id"))
                    if key not in seen:
                        seen.add(key)
                        all_results.append(r)
            
            return all_results
            
        except Exception as e:
            print("Triton error:", e)
            return []
