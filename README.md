# Fashion Image Retrieval System

## Overview

This project implements a multimodal fashion image retrieval system that retrieves visually relevant fashion images based on natural language queries.

The system uses:

- OpenCLIP (ViT-H-14) for multimodal image-text embeddings
- BLIP Image Captioning for semantic descriptions
- FAISS for efficient vector similarity search
- Metadata-aware re-ranking for improved fashion retrieval

---

## Features

- Natural language image search
- CLIP-based multimodal retrieval
- BLIP-generated captions
- Metadata-aware re-ranking
- Fast FAISS vector search
- Top-5 fashion image retrieval

---

## Project Structure

```
glance-fashion-retrieval/

├── app.py
├── requirements.txt
├── README.md

├── dataset/
│   └── images/

├── output/
│   ├── fashion.index
│   ├── captions.json
│   ├── metadata.json
│   ├── filenames.json
│   └── embeddings.npy

├── notebook/
│   ├── Indexer.ipynb
│   └── Retriever.ipynb

├── retriever/
│   └── search.py

├── utils/
│   ├── attribute_extractor.py
│   └── embedding.py

├── indexer/
│   └── build_index.py

└── report/
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python app.py
```

Example query:

```
Professional business attire inside a modern office
```

---

## Retrieval Pipeline

1. Generate BLIP captions.
2. Generate OpenCLIP embeddings.
3. Store embeddings in FAISS.
4. Encode user query.
5. Retrieve Top-50 candidates.
6. Re-rank using fashion metadata.
7. Return Top-5 images.

---

## Technologies Used

- Python
- PyTorch
- OpenCLIP
- BLIP
- FAISS
- NumPy
- Matplotlib
- Pillow

---

## Future Improvements

- Larger vision-language models
- Better attribute extraction
- ANN indexes for very large datasets
- Web interface using Streamlit
- Multi-language query support