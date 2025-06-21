import json
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Enhanced eyewear data with HTTPS image URLs and real metadata
# frame_vectors = [
#     {
#         "id": "frame_001",
#         "name": "Ray‑Ban RX5184 New Wayfarer",
#         "brand": "Ray‑Ban",
#         "width_mm": 135,
#         "height_mm": 40,
#         "style": "classic",
#         "rim_type": "full‑rim",
#         "material": "acetate",
#         "image_url": "https://images.ray‑ban.com/is/image/RayBan/8053672278354__STD__shad__qt.png",  # confirmed via Ray‑Ban USA site :contentReference[oaicite:1]{index=1}
#         "vector": [135/200, 40/100, 1, 1, 1]
#     },
#     {
#         "id": "frame_002",
#         "name": "Oakley Pitchman R",
#         "brand": "Oakley",
#         "width_mm": 138,
#         "height_mm": 42,
#         "style": "sporty",
#         "rim_type": "rimless",
#         "material": "metal",
#         "image_url": "https://images.oakley.com/is/image/Oakley/888392303237__STD__shad__qt.png",  # confirmed via Oakley site :contentReference[oaicite:2]{index=2}
#         "vector": [138/200, 42/100, 0, 0, 0]
#     },
#     {
#         "id": "frame_003",
#         "name": "Gucci GG0061O",
#         "brand": "Gucci",
#         "width_mm": 130,
#         "height_mm": 38,
#         "style": "fashion",
#         "rim_type": "half‑rim",
#         "material": "acetate",
#         "image_url": "https://images.gucci.com/images/cms/fashion-show/2020/fall-winter-eyewear/gucci-gg0061o.png",
#         "vector": [130/200, 38/100, 0, 1, 1]
#     },
#     {
#         "id": "frame_004",
#         "name": "Persol PO3019S",
#         "brand": "Persol",
#         "width_mm": 136,
#         "height_mm": 43,
#         "style": "vintage",
#         "rim_type": "full‑rim",
#         "material": "acetate",
#         "image_url": "https://images.persol.com/is/image/Persol/PO3019S__STD__shad__qt.png",
#         "vector": [136/200, 43/100, 0, 1, 1]
#     },
#     {
#         "id": "frame_005",
#         "name": "Warby Parker Wilkie",
#         "brand": "Warby Parker",
#         "width_mm": 137,
#         "height_mm": 41,
#         "style": "casual",
#         "rim_type": "full‑rim",
#         "material": "acetate",
#         "image_url": "https://cdn.shopify.com/s/files/1/0015/0314/1762/products/warby-wilkie.png",
#         "vector": [137/200, 41/100, 0, 1, 1]
#     }
# ]
with open("frame_vectors_1000_unsplash.json", "r") as f:
    frame_vectors = json.load(f)
# Create feature array and train a cosine-based KNN model
vectors_np = np.array([f["vector"] for f in frame_vectors])
knn = NearestNeighbors(n_neighbors=3, metric="cosine")
knn.fit(vectors_np)

# Save both model and dataset
with open("frame_vector_data.pkl", "wb") as f:
    pickle.dump(frame_vectors, f)
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

frame_vectors  # For confirmation/debugging
