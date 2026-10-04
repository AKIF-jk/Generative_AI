from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import time
import os

router = APIRouter(prefix="/api/face-to-sketch", tags=["face-to-sketch"])

# Must stay in sync with src/task4_face2sketch/data/augmentations.py (RESAMPLE).
# This Docker image only copies backend/app, so it cannot import that module.
RESAMPLE = Image.Resampling.BICUBIC

class ONNXModelSession:
    def __init__(self):
        self.session = None
        self.model_path = os.environ.get("ONNX_MODEL_PATH", "generator.onnx")

    def load_model(self):
        if self.session is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file {self.model_path} not found.")
            self.session = ort.InferenceSession(self.model_path)
            print("Loaded ONNX model.")

    def run(self, photo_array, style_id):
        self.load_model()
        inputs = {
            self.session.get_inputs()[0].name: photo_array,
            self.session.get_inputs()[1].name: style_id
        }
        return self.session.run(None, inputs)[0]

model_session = ONNXModelSession()

@router.on_event("startup")
async def startup_event():
    # Attempt to preload if the model exists, otherwise wait until inference
    try:
        model_session.load_model()
    except FileNotFoundError:
        print("ONNX model not found on startup, will attempt later.")

@router.post("")
async def generate_sketch(photo: UploadFile = File(...), style: int = Form(...)):
    if style not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="Style must be 1, 2, or 3")
    
    # Read and validate image
    contents = await photo.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    start_time = time.time()
    
    # Preprocess
    img = img.resize((128, 128), RESAMPLE)
    img_array = np.array(img).astype(np.float32)
    # Normalize to [-1.0, 1.0]
    img_array = (img_array / 127.5) - 1.0
    # To NCHW
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Style mapping (1, 2, 3 -> 0, 1, 2)
    style_idx = np.array([style - 1], dtype=np.int64)
    
    try:
        out_array = model_session.run(img_array, style_idx)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Please ensure generator.onnx is downloaded.")
        
    # Postprocess [-1.0, 1.0] -> [0, 255]
    out_img = out_array[0] # CHW
    out_img = (out_img * 0.5 + 0.5) * 255.0
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    out_img = np.transpose(out_img, (1, 2, 0))
    
    result_img = Image.fromarray(out_img)
    
    # Save to buffer
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    
    inference_time_ms = int((time.time() - start_time) * 1000)
    
    headers = {
        "X-Inference-Time-ms": str(inference_time_ms),
        "X-Model-Version": "v1.0-onnx"
    }
    
    return Response(content=buf.getvalue(), media_type="image/png", headers=headers)
