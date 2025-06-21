# prepare_embeddings.py
import json
import requests
import numpy as np
import cv2
import os
import pickle
from insightface.app import FaceAnalysis
from sklearn.neighbors import NearestNeighbors

# Example eyewear data (you can expand this)
with open("frame_vectors_1000_unsplash.json", "r") as f:
    frame_vectors = json.load(f)

# Initialize InsightFace
face_embedder = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0, det_size=(640, 640))

# Create output vectors
filtered_vectors = []
for frame in frame_vectors:
    try:
        resp = requests.get(frame["image_url"], timeout=10)
        img_arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        faces = face_embedder.get(img)
        if not faces:
            print(f"[WARN] No face found in {frame['id']}")
            continue

        emb = faces[0].embedding
        frame["vector"] = emb.tolist()
        filtered_vectors.append(frame)
        print(f"[INFO] Processed {frame['id']}")
    except Exception as e:
        print(f"[ERROR] Failed {frame['id']}: {e}")

# Train KNN
vectors_np = np.array([f["vector"] for f in filtered_vectors])
knn_model = NearestNeighbors(n_neighbors=3, metric="cosine")
knn_model.fit(vectors_np)

# Save
with open("frame_vector_data.pkl", "wb") as f:
    pickle.dump(filtered_vectors, f)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)

print(f"[DONE] Saved {len(filtered_vectors)} frame embeddings.")
