import faiss
import json
import os

from utils.embedding import encode_query
from utils.attribute_extractor import parse_query

def load_resources(output_dir):

    index = faiss.read_index(
        os.path.join(output_dir, "fashion.index")
    )

    with open(os.path.join(output_dir, "filenames.json")) as f:
        filenames = json.load(f)

    with open(os.path.join(output_dir, "metadata.json")) as f:
        metadata = json.load(f)

    return index, filenames, metadata

def retrieve_candidates(
    query,
    index,
    filenames,
    metadata,
    model,
    tokenizer,
    device,
    top_k=50
):

    embedding = encode_query(
        query,
        model,
        tokenizer,
        device
    )

    scores, indices = index.search(
        embedding,
        top_k
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        filename = filenames[idx]

        candidates.append({
            "filename": filename,
            "score": float(score),
            "metadata": metadata[filename]
        })

    return candidates


def rerank(candidates, query_info):

    for item in candidates:

        bonus = 0

        meta = item["metadata"]

        # Scene matching
        if query_info["scene"] != "unknown":
            if meta["scene"] == query_info["scene"]:
                bonus += 0.20

        # Style matching
        if query_info["style"] != "unknown":
            if meta["style"] == query_info["style"]:
                bonus += 0.15

        # Clothing overlap
        clothing_overlap = set(query_info["clothing"]) & set(meta["clothing"])
        bonus += 0.05 * len(clothing_overlap)

        # Color overlap
        color_overlap = set(query_info["colors"]) & set(meta["colors"])
        bonus += 0.05 * len(color_overlap)

        item["final_score"] = item["score"] + bonus

    candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return candidates


def search(
    query,
    index,
    filenames,
    metadata,
    model,
    tokenizer,
    device,
    top_k=5
):

    candidates = retrieve_candidates(
        query=query,
        index=index,
        filenames=filenames,
        metadata=metadata,
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=50
    )

    query_info = parse_query(query)

    results = rerank(candidates, query_info)

    return results[:top_k]


import matplotlib.pyplot as plt
from PIL import Image


def show_results(results, image_dir):

    plt.figure(figsize=(18, 8))

    for i, item in enumerate(results):

        image_path = os.path.join(image_dir, item["filename"])

        img = Image.open(image_path)

        plt.subplot(1, len(results), i + 1)
        plt.imshow(img)
        plt.axis("off")

        plt.title(
            f"{item['final_score']:.2f}\n{item['metadata']['caption']}",
            fontsize=8
        )

    plt.tight_layout()
    plt.show()