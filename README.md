# RAG With a Real Evaluation Harness

**Local RAG over the FastAPI docs that answers with cited sources - and an eval harness that caught its own metric lying (recall@1 0.76, MRR 0.86 across a 2x2 chunker/embedder grid).**

Everyone demos RAG. Almost nobody evals it. This project was built eval-first:
a 70-question golden set with file-level ground truth, hand-written retrieval
metrics, an A/B grid of two chunking strategies x two embedding models, and a
written failure analysis. The chat is the by-product; the harness is the
point.

![demo](docs/demo.gif)

## What's inside

```
FastAPI docs (fetched at pinned tag 0.139.2, never committed)
      | parse markdown, curate (changelog & meta pages excluded)
      v
chunkers ---- fixed 800/150 --+-- all-MiniLM-L6-v2 --+
        +---- heading-based --+                      +-- 4 ChromaDB collections
                              +-- bge-small-en-v1.5 -+
      v
retrieval (top-k, cosine) -> prompt with numbered citations -> llama3.1:8b (Ollama)
      |
      +-- rag ask / rag eval / rag ingest / rag fetch   (CLI)
      +-- streamlit app                                  (demo)
      +-- eval harness: recall@1/3/5/10 + MRR over distinct files,
          per-question miss reports, LLM-as-judge answer scoring
```

Plain Python on purpose - no LangChain, no LlamaIndex. Every step (chunking,
embedding, retrieval, prompting, metrics) is ~a screen of code you can read
and I can defend in an interview.

## Results

70 golden questions, ranks computed over distinct retrieved files
([why that matters](docs/eval-report.md#how-the-metrics-are-computed-and-the-bug-we-caught)):

| config | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| fixed__minilm | 0.657 | 0.871 | 0.943 | 0.957 | 0.775 |
| **fixed__bge** | **0.757** | **0.957** | **0.971** | **1.000** | **0.861** |
| sections__minilm | 0.671 | 0.857 | 0.914 | 1.000 | 0.786 |
| sections__bge | 0.729 | 0.929 | 0.971 | 1.000 | 0.838 |

Headline findings (full analysis in [docs/eval-report.md](docs/eval-report.md)):

- **The embedding model dominates; the chunker barely matters.** Swapping
  MiniLM -> bge-small buys +6 to +10 points of recall@1; the "smarter"
  heading-based chunker loses narrowly to plain fixed windows on the winning
  embedder.
- **The first version of the metric was wrong** - it ranked chunk slots, so
  overlapping chunks let one chunker win recall@5 by duplicate accounting.
  Distinct-file ranking erased that "win". Computing a metric precisely is not
  the same as measuring the right thing.
- **LLM-judged answer quality: 77% (~85% after manually auditing the judge's
  own mistakes)** - with documented same-model bias and an 11-point spread
  across three near-identical judge setups.

## Quickstart

Prereqs: Python 3.11+, [Ollama](https://ollama.com) with `ollama pull llama3.1:8b`.

```bash
python -m venv .venv && .venv/Scripts/activate    # or source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

rag fetch          # download the docs corpus (pinned tag)
rag ingest         # chunk + embed + index all 4 configs (~10 min on CPU)
rag ask "How do I declare optional query parameters?"
rag eval           # the point of the project
streamlit run app.py
```

Everything runs locally: no API keys, no cost, reproducible end to end.

## Honest evaluation, by construction

- Golden set authored from the docs *before* any config was compared, tagged
  with file-level truth so no chunking strategy is favored by the labels;
  every pair independently verified against its source file.
- The eval refuses to run against stale or partially rebuilt indexes (ingest
  manifest check) - the four configs are always compared on the same corpus.
- Metrics are hand-written pure functions with unit tests (`recall@k`, `MRR`),
  not framework numbers.
- The judge's score ships with its caveats attached, including the 11-point
  swing across three reasonable judge setups.
- 66 pytest tests cover chunkers, metrics, parsing, storage, prompts, the
  golden-set loader, and a pure-ASCII policy for every tracked file.

## Limitations

One corpus, one language, single-truth labels that undercount sibling-page
near-hits, golden questions that share vocabulary with their sources, and a
noisy same-model judge - all quantified or discussed in the
[eval report](docs/eval-report.md#limitations).

## Data & licensing

Code is [MIT](LICENSE). The corpus (FastAPI documentation, (c) Sebastian
Ramirez, MIT) is downloaded at build time and never committed. The golden
set's reference answers in `data/golden_set.jsonl` are condensed from that
MIT-licensed documentation at tag 0.139.2 - attribution kept here on purpose.

<!--
GitHub About: Eval-first local RAG over the FastAPI docs: 70-question golden set, recall@k/MRR across a 2x2 grid, MRR 0.86. No LangChain.
Topics: rag, llm-evaluation, retrieval-augmented-generation, embeddings, vector-search, chromadb, sentence-transformers, ollama, python, fastapi
-->
