-- Runs once when Docker creates a new volume; the runner also executes this idempotently.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS day6_chunks (
    strategy text NOT NULL,
    chunk_id text NOT NULL,
    document_id text NOT NULL,
    source text NOT NULL,
    page integer NOT NULL,
    text text NOT NULL,
    metadata jsonb NOT NULL,
    embedding vector(768) NOT NULL,
    PRIMARY KEY (strategy, chunk_id)
);
