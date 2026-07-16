# Fashion Image Retrieval System

## Problem Statement

Develop a multimodal fashion retrieval system capable of retrieving relevant fashion images from natural language queries.

---

## Proposed Solution

The proposed solution combines image understanding and text understanding using modern vision-language models.

Pipeline:

Images
↓

BLIP Caption Generation

↓

OpenCLIP Image & Text Embeddings

↓

FAISS Vector Index

↓

Natural Language Query

↓

Candidate Retrieval

↓

Metadata-aware Re-ranking

↓

Top-5 Results

---

## Models Used

### OpenCLIP

Used for generating image and text embeddings in a shared semantic space.

### BLIP

Used for generating captions for each image.

### FAISS

Used for efficient nearest-neighbor vector search.

---

## Improvements over Vanilla CLIP

The baseline CLIP retrieval is enhanced using metadata-aware re-ranking.

Extracted metadata includes:

- Clothing
- Colors
- Style
- Scene

These attributes improve retrieval quality for fashion-specific queries.

---

## Trade-offs

Advantages

- Fast retrieval
- No model training required
- Modular architecture
- Scalable indexing

Limitations

- Rule-based attribute extraction
- Caption quality depends on BLIP
- Scene recognition can be improved

---

## Scalability

FAISS enables efficient retrieval even for datasets containing millions of images.

Image indexing is performed offline while only the query is encoded during retrieval.

---

## Future Work

- LLM-based attribute extraction
- Better fashion ontology
- Approximate nearest-neighbor search
- User relevance feedback
- Web deployment