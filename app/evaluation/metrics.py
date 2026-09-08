import json
import re

from app.services.qa_service import get_groq_client
from app.config import GROQ_MODEL_NAME


def retrieval_precision_recall(retrieved_pages, expected_pages):
    """Precision: of what we retrieved, how much was actually relevant?
    Recall: of what was actually relevant, how much did we retrieve?
    """
    if not expected_pages:
        return None, None

    retrieved_set = set(retrieved_pages)
    expected_set = set(expected_pages)

    if not retrieved_set:
        return 0.0, 0.0

    true_positives = retrieved_set & expected_set
    precision = len(true_positives) / len(retrieved_set)
    recall = len(true_positives) / len(expected_set)
    return precision, recall


def keyword_coverage(answer: str, expected_keywords: list):
    """Simple, cheap sanity check: what fraction of expected keywords
    actually appear in the answer text (case-insensitive)?"""
    if not expected_keywords:
        return None

    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def llm_judge(question: str, answer: str, context: str, source_type: str = "document"):
    """Uses the LLM itself as a judge to score faithfulness (is the answer
    grounded in the context, not hallucinated?) and relevance (does the
    answer actually address the question?). Returns scores from 0 to 1.

    source_type matters here: a document-sourced answer should be judged
    against the retrieved document context, while a web-sourced answer should
    be judged against the web search results instead. Grading a web
    answer against document context would unfairly score it as
    unfaithful, since it was never meant to come from the document.
    """
    if source_type == "web":
        context_label = "Web search results provided to the AI"
        faithfulness_instruction = (
            "does the answer only make claims that are actually supported "
            "by these web search results? (1 = fully grounded in the "
            "search results, 0 = made things up beyond what they say)"
        )
    elif source_type == "none":
        context_label = "No context was available (neither document nor web)"
        faithfulness_instruction = (
            "since no context was available, a faithful answer should "
            "honestly say it could not find the information, rather than "
            "guessing. (1 = honestly admitted it doesn't know, 0 = "
            "invented an answer anyway)"
        )
    else:
        context_label = "Document context provided to the AI"
        faithfulness_instruction = (
            "does the answer only make claims that are actually supported "
            "by this document context? (1 = fully grounded, 0 = made "
            "things up)"
        )

    prompt = f"""You are evaluating an AI system's answer for quality.

Question: {question}

{context_label}:
{context if context else "(none provided)"}

AI's answer:
{answer}

Score the answer on two dimensions, each from 0 to 1:
- "faithfulness": {faithfulness_instruction}
- "relevance": does the answer actually address the question asked?
  (1 = directly relevant, 0 = off-topic or non-answer)

Respond with ONLY a JSON object like:
{{"faithfulness": 0.9, "relevance": 1.0}}"""

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return {"faithfulness": None, "relevance": None}