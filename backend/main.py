import base64
import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image
from video_detector import BoundingBox, process_frame, process_video
from ultralytics import YOLO
import uuid
import shutil
from fastapi import BackgroundTasks
from fastapi import HTTPException

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
TEMP_UPLOADS_DIR = "temp_uploads"
GENERATION_RESULTS_DIR = "generation_results"
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


def iter_file(path: str, start: int = 0, end: int = None, chunk_size: int = 1024*1024):
    """
    Generator to read a file in chunks from byte `start` to `end`.
    """
    with open(path, 'rb') as f:
        f.seek(start)
        remaining = (end - start + 1) if end is not None else None
        while True:
            read_size = chunk_size if remaining is None else min(
                chunk_size, remaining)
            data = f.read(read_size)
            if not data:
                break
            yield data
            if remaining:
                remaining -= len(data)
                if remaining <= 0:
                    break


@app.post("/process-video")
async def process_video_endpoint(
    model: str = Form(...),
    video: UploadFile = File(...)
):
    """
    Accepts a model name and a video, starts a background task for object detection,
    and returns a path to retrieve the processed video later.
    """
    yolo_model = get_model(model)
    if not yolo_model:
        return {"error": "Model not found"}, 404

    # Save uploaded video to a temporary location
    temp_video_filename = f"{uuid.uuid4()}-{video.filename}"
    temp_video_path = os.path.join(TEMP_UPLOADS_DIR, temp_video_filename)
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Prepare a unique path for the output video
    output_video_filename = f"processed-{temp_video_filename}"
    output_video_path = os.path.join(
        GENERATION_RESULTS_DIR, output_video_filename)

    process_video(video_path=temp_video_path,
                  model=yolo_model,
                  min_conf_level=0.4,
                  show_on_screen=False,
                  create_record=True,
                  output_path=output_video_path,
                  resW=854,
                  resH=480)
    # Return a URL-friendly path for the client to poll
    result_url_path = os.path.join(
        "generated-videos", output_video_filename).replace('\\', '/')
    os.remove(temp_video_path)
    return {"result_path": result_url_path}


@app.get("/generated-videos/{video_filename:path}")
async def get_generated_video(video_filename: str):
    """
    Streams a processed video file if it exists.
    """
    video_path = os.path.join(GENERATION_RESULTS_DIR, video_filename)

    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404, detail="Video not found or is still being processed.")

    # return FileResponse(video_path, media_type="video/mp4")
    headers = {
        "Content-Length": str(os.path.getsize(video_path)),
        "Accept-Ranges": "bytes"
    }
    return StreamingResponse(iter_file(video_path), media_type="video/mp4", headers=headers)


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
    if not os.path.exists(TEMP_UPLOADS_DIR):
        os.makedirs(TEMP_UPLOADS_DIR)
        print(f"Created directory: {TEMP_UPLOADS_DIR}")
    if not os.path.exists(GENERATION_RESULTS_DIR):
        os.makedirs(GENERATION_RESULTS_DIR)
        print(f"Created directory: {GENERATION_RESULTS_DIR}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
