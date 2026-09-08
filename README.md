# AI Document Assistant

A full-stack Retrieval-Augmented Generation (RAG) application that lets you upload documents (PDF, DOCX, TXT) and ask questions about them in natural language — with accurate, cited, streaming answers.

Built as a portfolio project to demonstrate production-oriented RAG engineering: hybrid retrieval, cross-encoder reranking, citation verification, and an automated evaluation harness — not just a basic "embed and ask" demo.

## Features

### Core
- **Multi-format document support** — PDF, DOCX, and TXT, with automatic format detection
- **Multi-document sessions** — upload several documents and switch between them
- **Streaming chat interface** — answers appear live, word by word
- **Conversation memory** — follow-up questions ("explain that more") understand prior context
- **Dark, custom-designed UI** — no template/starter styling

### Retrieval & Answer Quality
- **Hybrid search** — combines dense vector search (semantic similarity) with BM25 keyword search, so both conceptual matches and exact/rare terms are caught
- **Cross-encoder reranking** — a second-pass model (`ms-marco-MiniLM-L-6-v2`) re-scores candidate chunks against the exact question for more precise ranking than embedding similarity alone
- **Citation verification** — after an answer is generated, each cited source is checked against the answer to confirm it actually supports the claim, before being shown to the user
- **Direct page lookup** — "what is on page 12" retrieves that exact page by its printed page number (not just raw file position), bypassing similarity search entirely
- **Query rewriting** — vague or context-dependent questions are rewritten into standalone queries before being used for web fallback search
- **Web search fallback** — if a document genuinely cannot answer a question, the app automatically falls back to a live web search (via Tavily) and clearly labels the answer's source (document vs. web) in the UI

### Evaluation
- **Automated evaluation harness** (`app/evaluation/`) — runs a test suite of questions against the live retrieval + generation pipeline and scores:
  - Retrieval **precision** and **recall** against expected source pages
  - **Keyword coverage** in generated answers
  - **LLM-judged faithfulness** (is the answer grounded in its cited context, not hallucinated?) and **relevance**, with source-type-aware grading (document-grounded vs. web-grounded answers are judged against their correct context)
  - Full JSON reports saved per run for tracking changes over time

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| RAG / AI | LangChain, HuggingFace `sentence-transformers` (embeddings + cross-encoder), ChromaDB, `rank_bm25`, Groq (LLM inference) |
| Web search fallback | Tavily API |
| Frontend | Vanilla HTML / CSS / JavaScript, Jinja2 templating |
| Document parsing | `PyPDFLoader`, `Docx2txtLoader`, `TextLoader` |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  Browser (JS)   │────▶│  FastAPI Backend  │────▶│  Groq API             │
│  Streaming chat │     │  app/main.py      │     │  (LLM inference)      │
└─────────────────┘     └────────┬─────────┘     └───────────┬───────────┘
                                  │                           │
                        ┌─────────▼──────────────────────────▼─────────┐
                        │              app/services/qa_service.py       │
                        │   Hybrid retrieve → rerank → verify citations  │
                        │   → generate, with web-search fallback on      │
                        │   low-confidence document answers               │
                        └─────────┬──────────────────────┬──────────────┘
                                  │                       │
                   ┌──────────────▼─────────┐   ┌─────────▼──────────┐
                   │  ChromaDB + BM25        │   │  Tavily API         │
                   │  (semantic + keyword     │   │  (web fallback)     │
                   │   search over chunks)    │   └─────────────────────┘
                   └─────────────────────────┘
```

### How a question gets answered

1. If the question directly references a page number ("what is on page 12"), that exact page is looked up by its printed page number and answered directly — no search needed.
2. Otherwise, the question is run through **hybrid retrieval**: semantic search and BM25 keyword search both run, and their results are merged into one candidate pool.
3. The candidate pool is **reranked** with a cross-encoder, which scores the question against each candidate chunk directly for a more precise ranking.
4. The top chunks (plus the document's opening chunk, as a fallback for broad "what is this about" questions) are sent to the LLM with strict instructions to answer only from the provided context.
5. If the LLM indicates it cannot answer from the document, the app automatically retries using a **live web search**, with the question rewritten for clarity using recent conversation history if needed.
6. Before being shown, document-sourced citations are **verified** — the source is only kept if it genuinely supports a claim in the answer.
7. The final answer streams to the user, clearly labeled as document- or web-sourced.

## Project Structure

```
app/
├── api/            # Thin route handlers — HTTP in, HTTP out, no business logic
│   ├── pages.py     # Serves the frontend
│   ├── upload.py     # Document upload endpoint
│   ├── ask.py         # Question-answering endpoints (regular + streaming)
│   └── documents.py    # Lists uploaded documents for the switcher UI
├── services/       # All business logic lives here
│   ├── upload_services.py     # Saves uploaded files to disk
│   ├── pdf_processor.py         # Multi-format loading + chunking
│   ├── vector_store.py            # ChromaDB access, page-lookup helpers
│   ├── bm25_service.py              # Keyword search
│   ├── reranker_service.py            # Cross-encoder reranking
│   ├── web_search_service.py            # Tavily web search fallback
│   └── qa_service.py                      # Orchestrates retrieval, reranking,
│                                             citation verification, and generation
├── evaluation/     # Standalone evaluation harness
│   ├── metrics.py     # Precision/recall, keyword coverage, LLM-as-judge scoring
│   ├── run_eval.py     # Test runner
│   └── test_questions.json  # Test set (mix of in-document and out-of-scope questions)
├── static/         # CSS and JS
├── templates/      # Jinja2 HTML
├── config.py       # Centralized settings, loaded from .env
└── main.py         # App entrypoint and router registration
```

## Setup

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/keys)
- A [Tavily API key](https://app.tavily.com) (for web search fallback)

### Installation

```bash
git clone https://github.com/AHaadi29/AI-Document-Assistant.git
cd AI-Document-Assistant

python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Run

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000`.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `POST` | `/upload` | Uploads a document (PDF/DOCX/TXT), chunks it, and stores it in the vector database |
| `POST` | `/ask` | Answers a question against a document, returning the full answer at once |
| `POST` | `/ask/stream` | Same as `/ask`, but streams the answer token by token |
| `GET` | `/documents` | Lists all documents currently stored, for the document-switcher UI |

`/ask` and `/ask/stream` accept a JSON body of `{ "question": str, "history": [...], "document": str }`, where `history` is a list of prior `{question, answer}` turns used for conversational context.

## Evaluation

```bash
python -m app.evaluation.run_eval
```

This runs the test set in `app/evaluation/test_questions.json` through the live pipeline and prints scored results, plus a saved JSON report under `app/evaluation/results/`.

**Baseline results** (3-question subset, run against an AWS technical whitepaper, covering an in-document factual question, an in-document question with no expected-page annotation, and a deliberately out-of-scope question to test web fallback):

| Metric | Score |
|---|---|
| Average precision | 1.0 |
| Average recall | 1.0 |
| Average keyword coverage | 0.75 |
| Average faithfulness | 0.833 |
| Average relevance | 1.0 |

The out-of-scope question ("Who is the CEO of Tesla?") correctly triggered the web search fallback rather than hallucinating an answer from the unrelated document, and was scored against the web results it actually used rather than the document — this source-aware grading was a fix made after an initial run showed the naive approach unfairly penalizing correct web-sourced answers.

The test set has since been expanded to 8 questions covering page-lookup, plausible-but-absent questions, and multiple out-of-scope cases. Run the command above for current results.

## Known Limitations

- **Page lookup is PDF-specific.** DOCX and TXT files don't have a fixed concept of "printed page numbers," so page-lookup questions on those formats gracefully report that the page couldn't be found, rather than guessing.
- **Conversation history is client-side and in-memory.** Refreshing the page clears the current chat session; nothing is persisted server-side.
- **Single evaluation model.** The LLM-as-judge scoring uses the same model family that generates the answers, which can be less rigorous than an independent, stronger judge model — a common trade-off in lightweight eval setups.
- **Document upload replaces embeddings by filename**, not by content hash — re-uploading a file with the same name overwrites its stored chunks, which is intentional but worth knowing.

## Roadmap / possible future work

- Docker packaging for one-command setup
- Unit tests for core services
- Persistent chat sessions
- Confidence scoring surfaced in the UI

## License

MIT
