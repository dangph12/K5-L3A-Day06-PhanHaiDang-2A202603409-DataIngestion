"""Persistent pgvector storage and exact cosine search."""

import math
from pathlib import Path
from typing import Any, LiteralString, cast

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if type(dimensions) is not int or not 1 <= dimensions <= 16000:
            raise ValueError("dimensions must be an integer in 1..16000")
        self.database_url = database_url
        self.dimensions = dimensions
        self.backend_versions: dict[str, str] = {}

    def _vector(self, values: list[float]) -> str:
        if len(values) != self.dimensions:
            raise ValueError("Embedding dimension mismatch")
        if not all(math.isfinite(v) for v in values) or not any(values):
            raise ValueError("Cosine search requires finite, nonzero vectors")
        return "[" + ",".join(str(float(v)) for v in values) + "]"

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url) as conn:
            schema_path = Path(__file__).resolve().parents[1] / "init.sql"
            conn.execute(cast(LiteralString, schema_path.read_text()))
            dimension_row = conn.execute("""
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'day6_chunks'::regclass AND attname = 'embedding'
            """).fetchone()
            if dimension_row is None:
                raise RuntimeError("Embedding column is missing")
            dimension = dimension_row[0]
            version_row = conn.execute(
                "SELECT current_setting('server_version'), extversion "
                "FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if version_row is None:
                raise RuntimeError("pgvector extension is missing")
            self.backend_versions = dict(zip(("postgresql", "pgvector"), version_row))
            if dimension != self.dimensions:
                raise ValueError("Existing table uses another embedding dimension")

    def count(self, strategy: str) -> int:
        with psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                "SELECT count(*) FROM day6_chunks WHERE strategy = %s", (strategy,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Chunk count is missing")
            return row[0]

    def upsert(
        self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """Atomically replace a FULL strategy snapshot, removing stale chunks.

        Pass the complete corpus; [] clears a strategy. Repeated loads do not
        duplicate rows. All strategies are rebuilt with one model by the runner.
        """
        if len(chunks) != len(vectors):
            raise ValueError("Need exactly one vector per chunk")
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs")
        rows = []
        for chunk, vector in zip(chunks, vectors):
            metadata = {
                **chunk.metadata,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
            }
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Jsonb(metadata),
                    self._vector(vector),
                )
            )
        with psycopg.connect(self.database_url) as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (strategy,))
            conn.execute("DELETE FROM day6_chunks WHERE strategy = %s", (strategy,))
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO day6_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)
                """,
                    rows,
                )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        vector = self._vector(query_vector)
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, text, 1 - (embedding <=> %s::vector), metadata
                FROM day6_chunks
                WHERE strategy = %s AND metadata @> %s
                ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s

            """,
                (vector, strategy, Jsonb(filters or {}), vector, top_k),
            ).fetchall()
        return [
            {"chunk_id": r[0], "text": r[1], "score": float(r[2]), "metadata": r[3]}
            for r in rows
        ]
