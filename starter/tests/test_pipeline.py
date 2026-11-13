"""Focused invariants; no model download or database connection required."""
import json
import math
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from src.chunking import by_paragraph, fixed_size
from src.embeddings import MyEmbedder
from src.evaluation import evaluate_query, summarize
from src.models import DocumentPage
from src.run import (
    load_corpus,
    load_manifest,
    unwrap_questions,
    validate_questions,
    write_run_outputs,
)
from src.store import PgVectorStore


class EmbeddingTests(unittest.TestCase):
    def test_gemini_uses_distinct_retrieval_tasks_and_preserves_order(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as client:
            client.return_value.models.embed_content.side_effect = [
                SimpleNamespace(embeddings=[
                    SimpleNamespace(values=[1.0] * 768),
                    SimpleNamespace(values=[2.0] * 768),
                ]),
                SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 768)]),
            ]
            embedder = MyEmbedder()
            self.assertEqual(embedder.embed_documents(["first", "second"]),
                             [[1.0] * 768, [2.0] * 768])
            self.assertEqual(embedder.embed_query("question"), [0.5] * 768)
            calls = client.return_value.models.embed_content.call_args_list
            self.assertEqual(calls[0].kwargs["config"].task_type, "RETRIEVAL_DOCUMENT")
            self.assertEqual(calls[1].kwargs["config"].task_type, "RETRIEVAL_QUERY")
            self.assertEqual(calls[0].kwargs["contents"], ["first", "second"])
            self.assertEqual(calls[1].kwargs["contents"], "question")

    def test_gemini_rejects_incomplete_response(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as client:
            client.return_value.models.embed_content.return_value = SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0, 0.0])]
            )
            with self.assertRaises(ValueError):
                MyEmbedder().embed_documents(["one", "two"])

    def test_gemini_errors_propagate(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch(
            "google.genai.Client"
        ) as client:
            client.return_value.models.embed_content.side_effect = RuntimeError("quota")
            with self.assertRaisesRegex(RuntimeError, "quota"):
                MyEmbedder().embed_query("question")


class SchemaTests(unittest.TestCase):
    def test_schema_file_is_executed_and_existing_dimensions_are_preserved(self):
        with patch("src.store.psycopg.connect") as connect:
            connection = connect.return_value.__enter__.return_value
            connection.execute.return_value.fetchone.return_value = (384,)
            with self.assertRaisesRegex(ValueError, "another embedding dimension"):
                PgVectorStore("unused", 768).create_schema()
            statements = [call.args[0] for call in connection.execute.call_args_list]
            self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", statements[0])
            self.assertIn("embedding vector(768)", statements[0])
            self.assertFalse(any("DROP TABLE" in sql or "DELETE FROM" in sql
                                 for sql in statements))


class CorpusTests(unittest.TestCase):
    def test_manifest_converts_physical_page_keys(self):
        with patch(
            "src.run.read_json",
            return_value={"pages": {"17": {"chapter": "Statements"}}},
        ):
            corpus = load_manifest()
        self.assertEqual(corpus["pages"], {17: {"chapter": "Statements"}})

    def test_manifest_rejects_invalid_page_numbers(self):
        with (
            patch("src.run.read_json", return_value={"pages": {"17x": {}}}),
            self.assertRaisesRegex(ValueError, "manifest pages"),
        ):
            load_manifest()

    def test_unverified_corpus_fails_before_reading_pdf(self):
        corpus = load_manifest()
        corpus["title"] = ""
        with patch("src.run.load_pdf") as load_pdf:
            with self.assertRaisesRegex(ValueError, "verified corpus"):
                load_corpus(corpus)
            load_pdf.assert_not_called()


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.page = DocumentPage(
            "python_book", "https://example.org/book", 17,
            "First idea. Supporting passage. Last idea.",
            {"chapter": "Statements"},
        )
        self.entry = {
            "page": {
                "document_id": "python_book",
                "source": "https://example.org/book",
                "page": 17,
                "chapter": "Statements",
                "text": "Supporting passage.",
            },
            "question": {
                "id": "q01",
                "gold_source_id": "python_book",
                "gold_page": 17,
                "evidence_quote": "Supporting passage.",
            },
        }

    def test_unwraps_verified_passage_without_using_it_as_corpus(self):
        self.assertEqual(
            unwrap_questions([self.entry], [self.page]), [self.entry["question"]]
        )

    def test_rejects_wrong_page_source_metadata_or_passage(self):
        for field, value in (
            ("document_id", "other"),
            ("source", "other"),
            ("page", 18),
            ("chapter", "Expressions"),
            ("text", "Fabricated passage."),
        ):
            with self.subTest(field=field):
                entry = {
                    **self.entry,
                    "page": {**self.entry["page"], field: value},
                }
                with self.assertRaisesRegex(ValueError, "Benchmark page"):
                    unwrap_questions([entry], [self.page])

    def test_rejects_quote_outside_passage(self):
        entry = {
            **self.entry,
            "question": {**self.entry["question"], "evidence_quote": "First idea."},
        }
        with self.assertRaisesRegex(ValueError, "Benchmark page"):
            unwrap_questions([entry], [self.page])


class ResultTests(unittest.TestCase):
    def test_five_files_keep_run_inputs_and_audit_evidence(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "pipeline.log").write_text("run finished", encoding="utf-8")
            inputs = {"corpus": {"source_id": "book"}, "benchmark": [{"id": "q01"}]}
            review = {
                "coherence": [{"chunks": [{"text": "sample"}], "score_0_to_2": None}]
            }
            traces = [{"hits": [{"text": "evidence", "score": 0.8}]}]
            write_run_outputs(
                run_dir,
                inputs=inputs,
                traces=traces,
                metrics={"A": {"dev": {"hit_at_1": {"count": 1}}}},
                manual_review=review,
                run_info={"model": "gemini-embedding-001"},
            )
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {
                    "pipeline.log",
                    "run_info.json",
                    "traces.json",
                    "metrics.json",
                    "manual_review.json",
                },
            )
            self.assertEqual(
                json.loads((run_dir / "run_info.json").read_text())["inputs"], inputs
            )
            self.assertEqual(
                json.loads((run_dir / "manual_review.json").read_text()), review
            )
            self.assertEqual(json.loads((run_dir / "traces.json").read_text()), traces)


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.page = DocumentPage("book", "book.pdf", 7,
                                 "First idea.\n\nSecond idea.\n\n" + "long " * 20,
                                 {"chapter": "Basics"})

    def test_fixed_overlap_and_offsets(self):
        chunks = fixed_size(self.page, 20, 5)
        self.assertEqual(chunks[0].text[-5:], chunks[1].text[:5])
        for chunk in chunks:
            self.assertEqual(chunk.text, self.page.text[
                chunk.metadata["start_char"]:chunk.metadata["end_char"]])
            self.assertEqual(chunk.metadata["chapter"], "Basics")
            self.assertEqual((chunk.document_id, chunk.source, chunk.page),
                             ("book", "book.pdf", 7))

    def test_stable_ids_and_no_metadata_alias(self):
        first = fixed_size(self.page, 20, 5)
        second = fixed_size(self.page, 20, 5)
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])
        first[0].metadata["chapter"] = "Changed"
        self.assertEqual(self.page.metadata["chapter"], "Basics")

    def test_invalid_window(self):
        for size, overlap in ((0, 0), (5, 5), (5, -1), (5, 6)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                fixed_size(self.page, size, overlap)

    def test_paragraph_limit_and_complete_nonwhitespace_content(self):
        chunks = by_paragraph(self.page, 25)
        self.assertTrue(all(len(c.text) <= 25 for c in chunks))
        self.assertEqual("".join("".join(c.text.split()) for c in chunks),
                         "".join(self.page.text.split()))

    def test_paragraphs_packed_when_they_fit(self):
        page = DocumentPage("d", "s", 1, "One.\n\nTwo.")
        self.assertEqual(by_paragraph(page, 20)[0].text, page.text)

    def test_long_unbroken_text_is_preserved(self):
        page = DocumentPage("d", "s", 1, "a" * 101)
        self.assertEqual([len(c.text) for c in by_paragraph(page, 30)],
                         [30, 30, 30, 11])

    def test_empty_pages(self):
        page = DocumentPage("d", "s", 1, " \n\n ")
        self.assertEqual(fixed_size(page, 20, 0), [])
        self.assertEqual(by_paragraph(page, 20), [])
        with self.assertRaises(ValueError):
            by_paragraph(page, 0)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.question: dict[str, Any] = {"id": "q", "split": "test", "question": "What?", "expected_answer": "Full",
                             "gold_source_id": "book", "gold_page": 7,
                             "evidence_quote": "Full evidence with condition."}

    def hit(self, text, document_id="book", page=7):
        return {"chunk_id": "chunk", "text": text, "score": 0.8,
                    "metadata": {"document_id": document_id, "source": "book.pdf",
                                  "page": page, "chapter": "Basics"}}

    def test_requires_complete_evidence_and_correct_page(self):
        for hit in (self.hit("Full evidence"), self.hit(self.question["evidence_quote"], page=8),
                    self.hit(self.question["evidence_quote"], document_id="other")):
            self.assertFalse(evaluate_query(self.question, [hit])["hit_at_3"])

    def test_alternative_gold_and_top_three_limit(self):
        self.question["gold_alternatives"] = [
            {"gold_source_id": "other", "gold_page": 3, "evidence_quote": "Alternative."}]
        result = evaluate_query(self.question, [
            self.hit("No"), self.hit("Alternative.", "other", 3)])
        self.assertFalse(result["hit_at_1"])
        self.assertTrue(result["hit_at_3"])
        misses = [self.hit("No")] * 3
        self.assertFalse(evaluate_query(
            self.question, misses + [self.hit("Alternative.", "other", 3)])["hit_at_3"])

    def test_split_metrics_and_na_filter(self):
        trace = evaluate_query(self.question, [self.hit(self.question["evidence_quote"])])
        trace["strategy"] = "A"
        summary = summarize([trace])["A"]
        self.assertEqual(summary["test"]["hit_at_3"],
                         {"hits": 1, "total": 1, "rate": 1})
        self.assertIsNone(summary["dev"]["hit_at_1"]["rate"])
        self.assertIsNone(summary["test"]["filter_compliance"]["rate"])

    def test_empty_filtered_result_does_not_count_as_compliant(self):
        self.question["filter"] = {"chapter": "Basics"}
        trace = evaluate_query(self.question, [])
        trace["strategy"] = "C"
        self.assertEqual(summarize([trace])["C"]["test"]["filter_compliance"]["hits"], 0)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.pages = [DocumentPage("book", "book.pdf", 1, "Verified evidence.",
                                   {"chapter": "Basics"})]
        self.questions = [
            {"id": f"q{i}", "split": "dev" if i < 15 else "test",
                 "question": "Question?", "expected_answer": "Answer",
                 "gold_source_id": "book", "gold_page": 1, "evidence_quote": "Verified evidence.",
                 "filter": {"chapter": "Basics"} if i in (0, 1, 15) else {}}
            for i in range(20)]

    def test_accept_valid_benchmark(self):
        validate_questions(self.questions, self.pages)

    def test_reject_invalid_evidence(self):
        self.questions[0]["evidence_quote"] = "Invented."
        with self.assertRaises(ValueError):
            validate_questions(self.questions, self.pages)

    def test_reject_gold_source_filter(self):
        self.questions[0]["filter"] = {"document_id": "book"}
        with self.assertRaises(ValueError):
            validate_questions(self.questions, self.pages)

    def test_reject_insufficient_test_or_no_filter_test(self):
        with self.assertRaises(ValueError):
            validate_questions(self.questions[:19], self.pages)
        self.questions[15]["filter"] = {}
        with self.assertRaises(ValueError):
            validate_questions(self.questions, self.pages)

    def test_vector_validation_before_database_write(self):
        store = PgVectorStore("unused", 2)
        invalid_vectors: list[list[float]] = [
            [1.0], [0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0]
        ]
        for vector in invalid_vectors:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                store._vector(vector)
        with patch("src.store.psycopg.connect") as connect:
            with self.assertRaises(ValueError):
                store.upsert("A", [], [[1, 0]])
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
