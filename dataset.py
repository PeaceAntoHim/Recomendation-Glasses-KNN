import json
import random

brands = ["Ray-Ban","Oakley","Gucci","Persol","Warby Parker","Owndays","Gentle Monster","Lacoste","Tom Ford","Cartier"]
styles = ["classic","sporty","fashion","vintage","casual","minimalist"]
rim_types = ["full-rim","half-rim","rimless"]
materials = ["acetate","metal","plastic","titanium","ultem","TR-90"]

def unsplash_url():
    return "https://source.unsplash.com/320x240/?eyewear,glasses"

def encode_vector(width, height, style, rim_type, material):
    return [
        round(width/200,4),
        round(height/100,4),
        int(style=="classic"),
        int(rim_type=="full-rim"),
        int(material=="acetate")
    ]

frames = []
for i in range(1, 1001):
    w = random.randint(120,150)
    h = random.randint(35,50)
    style = random.choice(styles)
    rim = random.choice(rim_types)
    mat = random.choice(materials)
    brand = random.choice(brands)

    frames.append({
        "id": f"frame_{i:04}",
        "name": f"{brand} Model {i}",
        "brand": brand,
        "width_mm": w, "height_mm": h,
        "style": style, "rim_type": rim,
        "material": mat,
        "image_url": unsplash_url(),
        "vector": encode_vector(w, h, style, rim, mat)
    })

with open("frame_vectors_1000_unsplash.json", "w") as f:
    json.dump(frames, f, indent=2)

print("✅ 1000 frames generated with Unsplash images")
