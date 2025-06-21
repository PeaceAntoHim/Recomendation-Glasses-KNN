from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
import mediapipe as mp
import pickle
import os
from sklearn.neighbors import NearestNeighbors

app = FastAPI()
templates = Jinja2Templates(directory="templates")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Safely load data and model
if os.path.exists("frame_vector_data.pkl") and os.path.exists("knn_model.pkl"):
    with open("frame_vector_data.pkl", "rb") as f:
        frame_vectors = pickle.load(f)
    with open("knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
    frame_matrix = np.array([item["vector"] for item in frame_vectors])
else:
    frame_vectors = []
    knn_model = None
    frame_matrix = None

def classify_face_shape(landmarks):
    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Extract key points
    forehead = [landmarks[10].x, landmarks[10].y]
    chin = [landmarks[152].x, landmarks[152].y]
    left_jaw = [landmarks[234].x, landmarks[234].y]
    right_jaw = [landmarks[454].x, landmarks[454].y]
    left_temple = [landmarks[127].x, landmarks[127].y]
    right_temple = [landmarks[356].x, landmarks[356].y]

    # Metrics
    face_length = dist(forehead, chin)
    jaw_width = dist(left_jaw, right_jaw)
    forehead_width = dist(left_temple, right_temple)

    # Ratios
    face_ratio = jaw_width / face_length
    forehead_jaw_diff = forehead_width - jaw_width

    # Classification logic
    if abs(forehead_jaw_diff) < 0.01 and face_ratio > 0.8:
        return "Round"
    elif abs(forehead_jaw_diff) < 0.01 and face_ratio <= 0.8:
        return "Square"
    elif forehead_jaw_diff > 0.03:
        return "Heart"
    elif jaw_width > forehead_width and face_ratio < 0.7:
        return "Diamond"
    elif face_length / jaw_width > 1.5:
        return "Oblong"
    else:
        return "Oval"
    
def encode_user_vector(width, height, style, rim_type, material):
    return [
        round(width / 200, 4),
        round(height / 100, 4),
        int(style == "classic"),
        int(rim_type == "full-rim"),
        int(material == "acetate")
    ]

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze-face")
async def analyze_face(
    file: UploadFile = File(...),
    style: str = Form("classic"),
    rim_type: str = Form("full-rim"),
    material: str = Form("acetate")
):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0].landmark
    p1 = np.array([landmarks[234].x, landmarks[234].y])
    p2 = np.array([landmarks[454].x, landmarks[454].y])
    p3 = np.array([landmarks[10].x, landmarks[10].y])
    p4 = np.array([landmarks[152].x, landmarks[152].y])

    face_width = np.linalg.norm(p2 - p1)
    face_height = np.linalg.norm(p4 - p3)
    face_shape = classify_face_shape(landmarks)

    user_vector = encode_user_vector(
        face_width * 200,
        face_height * 100,
        style,
        rim_type,
        material
    )

    distances, indices = knn_model.kneighbors([user_vector])
    recommended = []
    for idx, dist in zip(indices[0], distances[0]):
        frame = frame_vectors[idx]
        similarity = 1 - dist
        recommended.append({
            "name": frame["name"],
            "brand": frame["brand"],
            "image_url": frame["image_url"],
            "style": frame["style"],
            "rim_type": frame["rim_type"],
            "material": frame["material"],
            "width_mm": frame["width_mm"],
            "height_mm": frame["height_mm"],
            "similarity": float(similarity)
        })

    return {
        "face_shape": face_shape,
        "face_width": face_width,
        "face_height": face_height,
        "recommended_frames": recommended
    }


@app.post("/recommend-frames")
async def recommend_frames(
    style: str = Form(...),
    rim_type: str = Form(...),
    material: str = Form(...),
    lifestyle: str = Form(...),
    prescription: str = Form(...),
    face_width: float = Form(...),
    face_height: float = Form(...)
):
    if not knn_model:
        return {"error": "Model not loaded"}

    try:
        user_vector = encode_user_vector(face_width, face_height, style, rim_type, material)
        distances, indices = knn_model.kneighbors([user_vector])

        top_matches = []
        for idx, dist in zip(indices[0], distances[0]):
            frame = frame_vectors[idx]
            frame["similarity"] = 1.0 - dist
            top_matches.append(frame)
        print(top_matches[:5])
        return {"recommended_frames": top_matches[:5]}

    except Exception as e:
        return {"error": f"Backend error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
