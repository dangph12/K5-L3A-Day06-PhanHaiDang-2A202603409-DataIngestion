"""Gemini text embeddings for document retrieval."""

import os

from google import genai
from google.genai import types


class MyEmbedder:
    model_name = "gemini-embedding-001"
    dimensions = 768

    def __init__(self) -> None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=key)

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        if not texts:
            return []
        vectors = []
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch if task == "RETRIEVAL_DOCUMENT" else batch[0],
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=self.dimensions
                ),
            )
            if not response.embeddings or len(response.embeddings) != len(batch):
                raise ValueError("Gemini returned an incomplete embedding batch")
            for embedding in response.embeddings:
                if not embedding.values or len(embedding.values) != self.dimensions:
                    raise ValueError("Gemini returned an invalid embedding dimension")
                vectors.append(embedding.values)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question], "RETRIEVAL_QUERY")[0]
