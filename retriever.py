"""Catalog Retriever — FAISS-based semantic search over the SHL catalog."""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class CatalogRetriever:
    """Searches the SHL catalog using FAISS vector similarity."""

    def __init__(self, catalog_path="data/catalog.json", index_path="data/catalog.index"):
        print("Loading catalog and search index...")

        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print(f"Loaded {len(self.catalog)} assessments with search index.")

    def search(self, query, top_k=15):
        """Find the most relevant assessments for a search query."""
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, min(top_k, len(self.catalog)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.catalog):
                item = self.catalog[idx].copy()
                item["relevance_score"] = float(score)
                results.append(item)

        return results

    def get_by_name(self, name):
        """Find a specific assessment by name."""
        name_lower = name.lower()
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None

    def format_for_prompt(self, assessments):
        """Format assessments as text for the LLM prompt."""
        lines = []
        for item in assessments:
            line = f"- Name: {item['name']}"
            line += f" | URL: {item['url']}"
            line += f" | Type: {item.get('test_type', 'N/A')}"
            if item.get('description'):
                desc = item['description'][:200]
                line += f" | Description: {desc}"
            if item.get('duration'):
                line += f" | Duration: {item['duration']}"
            if item.get('job_levels'):
                line += f" | Job Levels: {item['job_levels']}"
            if item.get('remote_testing'):
                line += f" | Remote: {item['remote_testing']}"
            lines.append(line)
        return "\n".join(lines)
