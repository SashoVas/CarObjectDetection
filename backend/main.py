import base64
import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from video_detector import BoundingBox, process_frame
from ultralytics import YOLO

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Allows Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
# Caching models
MODELS_CACHE = {}


def get_model(model_name: str):
    if model_name not in MODELS_CACHE:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            return None
        MODELS_CACHE[model_name] = YOLO(model_path, task='detect')
    return MODELS_CACHE[model_name]


@app.get("/models")
async def get_models():
    """
    Scans the models directory and its subdirectories and returns a list of available .pt model files.
    """
    models = []
    if os.path.isdir(MODELS_DIR):
        for root, _, files in os.walk(MODELS_DIR):
            for file in files:
                if file.endswith(".pt"):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, MODELS_DIR)
                    # On Windows, relpath can use backslashes, so we replace them
                    models.append(relative_path.replace('\\', '/'))

    return models


@app.post("/detect")
async def detect(model: str = Form(...), image: UploadFile = File(...)):
    """
    Accepts a model name and an image, performs object detection,
    and returns the processed image with bounding boxes and their coordinates.
    """
    yolo_model = get_model(model)
    if not yolo_model:
        return {"error": "Model not found"}, 404

    # Read uploaded image
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Process the frame
    bounding_boxes = process_frame(
        frame, yolo_model, yolo_model.names, min_conf_level=0.4)

    # Encode processed image to base64
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Format bounding boxes
    boxes_formatted = [
        {
            "x": int(bb.xmin),
            "y": int(bb.ymin),
            "w": int(bb.xmax - bb.xmin),
            "h": int(bb.ymax - bb.ymin),
            "class_name": bb.class_name,
            "conf": float(bb.conf)
        }
        for bb in bounding_boxes
    ]

    return {"image": f"data:image/jpeg;base64,{img_base64}", "boxes": boxes_formatted}

if __name__ == "__main__":
    import uvicorn
    # Ensure models directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
