import json
import numpy as np
import triton_python_backend_utils as pb_utils
from deepface import DeepFace
import cv2
import base64

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.extract_faces(img_path=dummy_img, detector_backend="retinaface", enforce_detection=False)
            DeepFace.represent(img_path=dummy_img, detector_backend="skip", enforce_detection=False)
        except Exception as e:
            print("Initialization warning:", str(e).encode('ascii', 'ignore').decode('ascii'))

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                in_0 = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES")
                img_array = in_0.as_numpy()
                
                out_num_list = []
                out_box_list = []
                out_emb_list = []
                
                for b_idx in range(img_array.shape[0]):
                    # HTTP JSON API with BYTES might pass the base64 string directly in Python backend
                    # Let's decode it.
                    val = img_array[b_idx][0]
                    if isinstance(val, bytes):
                        try:
                            # Try to b64decode, if it fails, assume it's raw bytes
                            img_b = base64.b64decode(val)
                        except Exception:
                            img_b = val
                    else:
                        img_b = base64.b64decode(val)
                        
                    img_np = np.frombuffer(img_b, dtype=np.uint8)
                    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        out_num_list.append(np.array([0], dtype=np.int32))
                        out_box_list.append(np.empty((0, 4), dtype=np.float32))
                        out_emb_list.append(np.empty((0, 512), dtype=np.float32))
                        continue
                        
                    faces = DeepFace.extract_faces(
                        img_path=img,
                        detector_backend="retinaface",
                        enforce_detection=False,
                        align=True
                    )
                    
                    valid_faces = []
                    for f in faces:
                        fa = f.get("facial_area", {})
                        area = (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)
                        conf = f.get("confidence") or 0
                        if conf >= 0.5 and area >= 64*64:
                            valid_faces.append(f)
                    
                    valid_faces.sort(key=lambda x: (x.get("facial_area", {}).get("w", 0) * x.get("facial_area", {}).get("h", 0)), reverse=True)
                    valid_faces = valid_faces[:3]
                    
                    embeddings = []
                    bboxes = []
                    
                    for f in valid_faces:
                        face_img = f["face"]
                        emb = DeepFace.represent(
                            img_path=face_img,
                            detector_backend="skip",
                            enforce_detection=False
                        )[0]["embedding"]
                        
                        fa = f.get("facial_area", {})
                        bboxes.append([fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)])
                        
                        emb_np = np.array(emb, dtype=np.float32)
                        norm = np.linalg.norm(emb_np)
                        if norm > 0:
                            emb_np = emb_np / norm
                        embeddings.append(emb_np)
                    
                    num_faces = len(valid_faces)
                    out_num_list.append(np.array([num_faces], dtype=np.int32))
                    if num_faces == 0:
                        out_box_list.append(np.empty((0, 4), dtype=np.float32))
                        out_emb_list.append(np.empty((0, 512), dtype=np.float32))
                    else:
                        out_box_list.append(np.array(bboxes, dtype=np.float32))
                        out_emb_list.append(np.array(embeddings, dtype=np.float32))
                
                MAX_FACES = 3
                padded_box_list = []
                padded_emb_list = []
                for b, e in zip(out_box_list, out_emb_list):
                    pad_len = MAX_FACES - b.shape[0]
                    if pad_len > 0:
                        b = np.vstack([b, np.zeros((pad_len, 4), dtype=np.float32)])
                        e = np.vstack([e, np.zeros((pad_len, 512), dtype=np.float32)])
                    padded_box_list.append(b)
                    padded_emb_list.append(e)
                
                out_num_tensor = pb_utils.Tensor("NUM_FACES", np.array(out_num_list, dtype=np.int32))
                out_box_tensor = pb_utils.Tensor("BBOXES", np.array(padded_box_list, dtype=np.float32))
                out_emb_tensor = pb_utils.Tensor("EMBEDDINGS", np.array(padded_emb_list, dtype=np.float32))
                
                response = pb_utils.InferenceResponse(output_tensors=[out_num_tensor, out_box_tensor, out_emb_tensor])
                responses.append(response)
                
            except Exception as e:
                print("Error processing request:", e)
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e))))
                
        return responses
