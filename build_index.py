"""Build FAISS search index from the scraped catalog data."""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_index():
    """Build a FAISS index from catalog.json."""
    catalog_path = os.path.join("data", "catalog.json")
    index_path = os.path.join("data", "catalog.index")

    print("Loading catalog...")
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Found {len(catalog)} assessments.")

    texts = []
    for item in catalog:
        parts = [item["name"]]
        if item.get("description"):
            parts.append(item["description"])
        if item.get("test_type"):
            parts.append(f"Test type: {item['test_type']}")
        if item.get("job_levels"):
            parts.append(f"Job levels: {item['job_levels']}")
        if item.get("duration"):
            parts.append(f"Duration: {item['duration']}")
        if item.get("remote_testing"):
            parts.append(f"Remote testing: {item['remote_testing']}")
        text = ". ".join(parts)
        texts.append(text)

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    print(f"Index contains {index.ntotal} vectors of dimension {dimension}")


if __name__ == "__main__":
    build_index()
