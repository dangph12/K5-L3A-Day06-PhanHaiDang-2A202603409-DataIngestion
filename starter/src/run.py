"""Run the same corpus/model/questions through A, B, and C."""

import hashlib
import json
import logging
import os
import statistics
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, evidence_matches, gold_evidence, summarize
from .ingestion import load_pdf
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
BENCHMARK = ROOT.parent / "benchmark/question.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# Complete these from the PDF and its license before running an experiment.
CORPUS = {
    "source_id": "python_book",
    "title": "",
    "path": "data/Introduction_to_Python_Programming.pdf",
    "source_path_or_url": "",
    "license_or_permission": "",
    "accessed_at": "",
    "pages": {},  # Physical PDF page number -> verified metadata, e.g. chapter.
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus():
    """Select only verified useful physical pages and their source metadata."""
    selected = CORPUS["pages"]
    if (
        any(
            not CORPUS[key]
            for key in (
                "title",
                "source_path_or_url",
                "license_or_permission",
                "accessed_at",
            )
        )
        or len(selected) < 30
        or any(type(number) is not int or number < 1 for number in selected)
    ):
        raise ValueError(
            "Configure at least 30 verified corpus pages and source facts in run.py"
        )
    loaded = load_pdf(
        ROOT / CORPUS["path"],
        CORPUS["source_id"],
        CORPUS["source_path_or_url"],
        end_page=max(selected),
    )
    pages = [loaded[number - 1] for number in sorted(selected)]
    for page in pages:
        if not page.text.strip():
            raise ValueError(f"Empty content page: {page.document_id}/{page.page}")
        metadata = selected[page.page]
        if not isinstance(metadata, dict) or set(metadata) & {
            "document_id",
            "source",
            "page",
            "start_char",
            "end_char",
        }:
            raise ValueError("Corpus metadata may not override traceability fields")
        page.metadata.update(metadata)
    keys = set().union(*(page.metadata for page in pages))
    if not any(
        len(
            {
                json.dumps(p.metadata.get(key), sort_keys=True)
                for p in pages
                if key in p.metadata
            }
        )
        >= 2
        for key in keys
    ):
        raise ValueError(
            "Need a useful metadata field with at least two verified values"
        )
    return pages


def validate_questions(questions, pages):
    if not isinstance(questions, list):
        raise TypeError("Benchmark must be a JSON array")
    counts = Counter(q.get("split") for q in questions)
    if counts["dev"] < 15 or counts["test"] < 5 or set(counts) - {"dev", "test"}:
        raise ValueError("Benchmark requires at least 15 dev and 5 test questions")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    filtered = [q for q in questions if q.get("filter")]
    if len(filtered) < 3 or not any(q["split"] == "test" for q in filtered):
        raise ValueError("Need at least three filter questions, including one test")
    source_pages = {(p.document_id, p.page): p for p in pages}
    allowed_filters = set().union(*(p.metadata for p in pages))
    for question in questions:
        for key in ("id", "question", "expected_answer"):
            if not isinstance(question.get(key), str) or not question[key].strip():
                raise ValueError(f"Question requires nonblank {key}")
        filters = question.get("filter") or {}
        if (
            not isinstance(filters, dict)
            or set(filters) - allowed_filters
            or any(
                type(value) not in (str, int, float, bool) for value in filters.values()
            )
        ):
            raise ValueError("Filters must use verified corpus metadata fields")
        if filters and not any(
            all(p.metadata.get(k) == v for k, v in filters.items()) for p in pages
        ):
            raise ValueError(f"Filter has no matching source page: {question['id']}")
        eligible_gold = False
        for gold in gold_evidence(question):
            page = source_pages.get((gold["gold_source_id"], gold["gold_page"]))
            if page is None or not evidence_matches(page.text, gold["evidence_quote"]):
                raise ValueError(f"Gold evidence absent from source: {question['id']}")
            if all(page.metadata.get(k) == v for k, v in filters.items()):
                eligible_gold = True
        if not eligible_gold:
            raise ValueError(f"Filter excludes all gold evidence: {question['id']}")


def main():
    load_dotenv(ROOT / ".env")
    run_dir = ROOT / "results" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8"),
        ],
    )
    try:
        started = time.perf_counter()
        pages = load_corpus()
        questions = read_json(BENCHMARK)
        validate_questions(questions, pages)
        # Selection rule is recorded before retrieval. Human labels are never inferred.
        write_json(
            run_dir / "inputs.json",
            {
                "corpus": CORPUS,
                "benchmark": questions,
                "selection_rule": (
                    "Prefer dev Hit@3, then dev Hit@1, then fewer characters. "
                    "Report human grounding/coherence tradeoffs separately."
                ),
                "size": CHUNK_SIZE,
                "overlap": CHUNK_OVERLAP,
            },
        )
        write_json(run_dir / "source_pages.json", [asdict(p) for p in pages])
        embedder = MyEmbedder()
        store = PgVectorStore(os.environ["DATABASE_URL"], embedder.dimensions)
        store.create_schema()
        # Embed each query once; no answer or gold metadata enters embedding input.
        queries = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
        traces, stats, coherence, grounding, metadata_review = [], {}, [], [], []
        paragraph_chunks = None
        paragraph_vectors = None
        # Positions are chosen independently of strategy scores.
        anchors = sorted(pages, key=lambda p: (p.document_id, p.page))[:10]
        for strategy in ("A", "B", "C"):
            ingest_start = time.perf_counter()
            if strategy == "A":
                chunks = [
                    c for p in pages for c in fixed_size(p, CHUNK_SIZE, CHUNK_OVERLAP)
                ]
                vectors = embedder.embed_documents([c.text for c in chunks])
            elif strategy == "B":
                chunks = [c for p in pages for c in by_paragraph(p, CHUNK_SIZE)]
                vectors = embedder.embed_documents([c.text for c in chunks])
                paragraph_chunks, paragraph_vectors = chunks, vectors
            else:
                assert paragraph_chunks is not None and paragraph_vectors is not None
                chunks, vectors = paragraph_chunks, paragraph_vectors
            store.upsert(strategy, chunks, vectors)
            first_count = store.count(strategy)
            reload_start = time.perf_counter()
            store.upsert(strategy, chunks, vectors)
            reloaded_count = store.count(strategy)
            if first_count != len(chunks) or reloaded_count != first_count:
                raise RuntimeError("Reload duplicated or lost chunks")
            reload_seconds = time.perf_counter() - reload_start
            lengths = [len(c.text) for c in chunks]
            stats[strategy] = {
                "chunks": len(chunks),
                "characters": sum(lengths),
                "reload_count": reloaded_count,
                "reload_seconds": reload_seconds,
                "length_min": min(lengths),
                "length_max": max(lengths),
                "length_mean": statistics.mean(lengths),
                "length_median": statistics.median(lengths),
                "ingest_seconds": time.perf_counter() - ingest_start,
                "embeddings_reused_from_B": strategy == "C",
                "embedding_api_cost": None,
            }
            write_json(run_dir / f"chunks_{strategy}.json", [asdict(c) for c in chunks])
            for page in anchors:
                position = next(i for i, ch in enumerate(page.text) if not ch.isspace())
                covering = [
                    asdict(c)
                    for c in chunks
                    if c.document_id == page.document_id
                    and c.page == page.page
                    and c.metadata["start_char"] <= position < c.metadata["end_char"]
                ]
                coherence.append(
                    {
                        "strategy": strategy,
                        "document_id": page.document_id,
                        "page": page.page,
                        "position": position,
                        "chunks": covering,
                        "score_0_to_2": None,
                        "reason": "",
                    }
                )
            for question in questions:
                filters = question.get("filter") if strategy == "C" else None
                hits = store.search(strategy, queries[question["id"]], filters=filters)
                trace = evaluate_query(question, hits)
                trace.update(strategy=strategy, applied_filter=filters or {})
                traces.append(trace)
                if question["split"] == "test":
                    grounding.append(
                        {
                            "strategy": strategy,
                            "question_id": question["id"],
                            "score_0_to_2": None,
                            "reason": "",
                        }
                    )
                    for rank, hit in enumerate(hits, 1):
                        metadata_review.append(
                            {
                                "strategy": strategy,
                                "question_id": question["id"],
                                "rank": rank,
                                "chunk_id": hit["chunk_id"],
                                "metadata": hit["metadata"],
                                "complete_and_correct": None,
                                "reason": "",
                            }
                        )
        metrics = summarize(traces)
        chosen = max(
            ("A", "B", "C"),
            key=lambda s: (
                metrics[s]["dev"]["hit_at_3"]["rate"],
                metrics[s]["dev"]["hit_at_1"]["rate"],
                -stats[s]["characters"],
            ),
        )
        write_json(run_dir / "traces.json", traces)
        write_json(run_dir / "metrics.json", metrics)
        write_json(
            run_dir / "manual_review.json",
            {
                "reviewer": "",
                "bias_and_method": "",
                "coherence": coherence,
                "grounding": grounding,
                "metadata": metadata_review,
                "instructions": (
                    "Verify against source_pages.json and original PDF. Fill labels "
                    "and reasons manually; null means unreviewed. Grounding measures "
                    "retrieved evidence, not LLM answer accuracy."
                ),
            },
        )
        write_json(
            run_dir / "run_info.json",
            {
                "model": embedder.model_name,
                "dimensions": embedder.dimensions,
                "provider": "Gemini API",
                "metric": "exact cosine",
                "backend_versions": store.backend_versions,
                "query_document_processing": (
                    "RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT; 768 dimensions"
                ),
                "versions": {
                    p: version(p)
                    for p in (
                        "google-genai",
                        "psycopg",
                        "pypdf",
                        "python-dotenv",
                    )
                },
                "benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
                "corpus_sha256": hashlib.sha256(
                    json.dumps(
                        [asdict(p) for p in pages], ensure_ascii=False, sort_keys=True
                    ).encode()
                ).hexdigest(),
                "statistics": stats,
                "dev_selected_strategy": chosen,
                "total_seconds": time.perf_counter() - started,
                "cost_note": (
                    "Gemini API usage is unpriced here; verify billed cost separately."
                ),
                "timing_note": (
                    "C reuses B embeddings; ingestion timings exclude PDF parsing "
                    "and shared model/query preparation."
                ),
                "manual_assessments_pending": True,
            },
        )
        LOG.info("Saved results to %s", run_dir)
    except Exception as error:
        # Avoid recording connection URLs or provider credentials in failure logs.
        LOG.error("Pipeline failed (%s); no fallback used.", type(error).__name__)
        raise


if __name__ == "__main__":
    main()
