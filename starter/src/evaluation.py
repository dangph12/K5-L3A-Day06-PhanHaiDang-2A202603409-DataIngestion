"""Evidence retrieval metrics; human judgments remain separate."""

from typing import Any

from .ingestion import normalize_text


def gold_evidence(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Alternatives are complete source/page/quote records in gold_alternatives."""
    return [question, *question.get("gold_alternatives", [])]


def evidence_matches(text: str, quote: str) -> bool:
    return bool(quote.strip()) and normalize_text(quote) in normalize_text(text)


def evaluate_query(
    question: dict[str, Any], hits: list[dict[str, Any]]
) -> dict[str, Any]:
    matches = []
    for hit in hits[:3]:
        meta = hit["metadata"]
        matches.append(
            any(
                meta.get("document_id") == gold["gold_source_id"]
                and meta.get("page") == gold["gold_page"]
                and evidence_matches(hit["text"], gold["evidence_quote"])
                for gold in gold_evidence(question)
            )
        )
    return {
        "question": question,
        "hits": hits[:3],
        "evidence_matches": matches,
        "hit_at_1": bool(matches and matches[0]),
        "hit_at_3": any(matches),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for strategy in sorted({trace["strategy"] for trace in traces}):
        result[strategy] = {}
        for split in ("dev", "test"):
            group = [
                t
                for t in traces
                if t["strategy"] == strategy and t["question"]["split"] == split
            ]
            count = len(group)
            metrics: dict[str, Any] = {"questions": count}
            for key in ("hit_at_1", "hit_at_3"):
                numerator = sum(t[key] for t in group)
                metrics[key] = {
                    "hits": numerator,
                    "total": count,
                    "rate": numerator / count if count else None,
                }
            filtered = [t for t in group if t["question"].get("filter")]
            compliant = sum(
                bool(t["hits"])
                and all(
                    all(
                        h["metadata"].get(k) == v
                        for k, v in t["question"]["filter"].items()
                    )
                    for h in t["hits"]
                )
                for t in filtered
            )
            metrics["filter_compliance"] = {
                "hits": compliant,
                "total": len(filtered),
                "rate": compliant / len(filtered) if filtered else None,
            }
            result[strategy][split] = metrics
    return result
