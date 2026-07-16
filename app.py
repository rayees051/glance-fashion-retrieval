import os
import torch
import open_clip

from retriever.search import (
    load_resources,
    search,
    show_results
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# -----------------------------
# Load OpenCLIP
# -----------------------------
print("Loading OpenCLIP...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14",
    pretrained="laion2b_s32b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-H-14")

model = model.to(device)
model.eval()

print("Model Loaded!")

# -----------------------------
# Load FAISS
# -----------------------------
print("Loading FAISS index...")

index, filenames, metadata = load_resources(OUTPUT_DIR)

print(f"Loaded {len(filenames)} images")

# -----------------------------
# Query Loop
# -----------------------------
while True:

    query = input("\nEnter your query (or type 'exit'): ")

    if query.lower() == "exit":
        break

    results = search(
        query=query,
        index=index,
        filenames=filenames,
        metadata=metadata,
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=5
    )

    show_results(results, IMAGE_DIR)