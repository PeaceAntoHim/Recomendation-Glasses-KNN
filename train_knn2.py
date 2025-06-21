import json
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Load eyewear frame data from JSON file
with open("frame_vectors_1000_unsplash.json", "r") as f:
    frame_vectors = json.load(f)

# Optional: Validate or normalize your vector data if needed
# Example vector format: [width_ratio, height_ratio, style_vector...]
vectors_np = np.array([f["vector"] for f in frame_vectors])

# Train KNN using cosine similarity
knn = NearestNeighbors(n_neighbors=3, metric="cosine")
knn.fit(vectors_np)

# Save the frame vector data
with open("frame_vector_data.pkl", "wb") as f:
    pickle.dump(frame_vectors, f)

# Save the trained KNN model
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

# Debug print (optional)
print(f"Loaded {len(frame_vectors)} frame vectors")
print("Sample:", frame_vectors[0])
