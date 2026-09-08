import json
from datetime import datetime
from pathlib import Path

from app.services.qa_service import answer_question
from app.services.vector_store import list_documents
from app.evaluation.metrics import retrieval_precision_recall, keyword_coverage, llm_judge

TEST_FILE = Path(__file__).resolve().parent / "test_questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def resolve_document_source(filename: str):
    """Turns a filename from the test set into the full source path
    that's actually stored in the vector database."""
    for doc in list_documents():
        if doc["filename"] == filename:
            return doc["source"]
    return None


def run_evaluation():
    test_cases = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    results = []

    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        expected_pages = case.get("expected_pages", [])
        expected_keywords = case.get("expected_keywords", [])
        source = resolve_document_source(case["document"])

        print(f"\n[{i}/{len(test_cases)}] {question}")

        if source is None:
            print(f"  SKIPPED: document '{case['document']}' not found in database.")
            continue

        result = answer_question(question, document=source)
        answer = result["answer"]
        sources = result.get("sources", [])
        source_type = result.get("source_type")

        retrieved_pages = [s["page"] for s in sources if "page" in s]
        precision, recall = retrieval_precision_recall(retrieved_pages, expected_pages)
        kw_score = keyword_coverage(answer, expected_keywords)

        if source_type == "web":
            context = "\n\n".join(f"{s.get('title', '')}: {s.get('url', '')}" for s in sources)
        else:
            context = "\n\n".join(s.get("snippet", "") for s in sources)

        judge_scores = llm_judge(question, answer, context, source_type=source_type)

        row = {
            "question": question,
            "answer": answer,
            "source_type": source_type,
            "retrieved_pages": retrieved_pages,
            "expected_pages": expected_pages,
            "precision": precision,
            "recall": recall,
            "keyword_coverage": kw_score,
            "faithfulness": judge_scores.get("faithfulness"),
            "relevance": judge_scores.get("relevance"),
        }
        results.append(row)

        print(f"  source_type: {source_type}")
        print(f"  precision: {precision}, recall: {recall}, keyword_coverage: {kw_score}")
        print(f"  faithfulness: {row['faithfulness']}, relevance: {row['relevance']}")

    print_summary(results)
    save_report(results)


def _average(values):
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def print_summary(results):
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total questions evaluated: {len(results)}")
    print(f"Average precision:         {_average([r['precision'] for r in results])}")
    print(f"Average recall:            {_average([r['recall'] for r in results])}")
    print(f"Average keyword coverage:  {_average([r['keyword_coverage'] for r in results])}")
    print(f"Average faithfulness:      {_average([r['faithfulness'] for r in results])}")
    print(f"Average relevance:         {_average([r['relevance'] for r in results])}")


def save_report(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"eval_{timestamp}.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()