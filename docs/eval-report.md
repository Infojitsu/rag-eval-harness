# Evaluation Report - RAG over the FastAPI Docs

**Date:** 2026-07-18
**Corpus:** FastAPI documentation, English (`docs/en/docs/`), pinned at git tag
`0.139.2` - 140 documents after curation (changelog and community/meta pages
excluded; the changelog alone would have added ~1,000 retrieval-noise chunks).
**Golden set:** 70 hand-authored question-answer pairs over 51 distinct files
(`data/golden_set.jsonl`), each tagged with the one corpus file that answers
it. Every pair was independently double-checked against its source file before
any evaluation ran.
**Hardware:** embeddings on CPU (sentence-transformers), generation and
judging on an RTX 4070 SUPER via Ollama (`llama3.1:8b`, temperature 0).

## The grid

Two chunking strategies x two embedding models = four indexed configs:

| | `all-MiniLM-L6-v2` | `bge-small-en-v1.5` |
|---|---|---|
| **fixed** (800 chars, 150 overlap) | `fixed__minilm` (1,112 chunks) | `fixed__bge` (1,112 chunks) |
| **sections** (heading-based, merge <200, split >2000) | `sections__minilm` (912 chunks) | `sections__bge` (912 chunks) |

Ground truth is **file-level**, so it is identical for every config - no
chunking strategy is favored by the labels.

## How the metrics are computed (and the bug we caught)

`recall@k` = fraction of questions whose ground-truth file appears in the top
k retrieved **distinct files**. `MRR` = mean reciprocal rank of the truth file
in that same distinct-file list. With one relevant document per question,
recall@k is the same thing as hit@k - we keep the standard name.

The first version of this harness ranked raw *chunk slots* instead of distinct
files. That silently favored the fixed chunker: its overlapping chunks put the
same file into several top-k slots, so with bge it "won" recall@5 by 2.9
points purely through duplicate accounting. With ranks computed over distinct
files, that gap collapses to an exact tie (0.971 vs 0.971). (The MiniLM pair
still shows a real 2.9-point recall@5 gap in the final table - same number,
different cause: that one survives deduplication.) The lesson is the whole
point of this project: **an eval can be precisely computed and still measure
the wrong thing.**

## Results (70 questions, distinct-file ranking)

| config | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| fixed__minilm | 0.657 | 0.871 | 0.943 | 0.957 | 0.775 |
| **fixed__bge** | **0.757** | **0.957** | **0.971** | **1.000** | **0.861** |
| sections__minilm | 0.671 | 0.857 | 0.914 | 1.000 | 0.786 |
| sections__bge | 0.729 | 0.929 | 0.971 | 1.000 | 0.838 |

### What actually mattered

1. **The embedding model dominates.** Swapping MiniLM -> bge-small is worth
   +6 to +10 points of recall@1 and +0.05 to +0.09 MRR, on both chunkers.
   bge-small is retrieval-tuned and gets its documented query prefix
   (`"Represent this sentence for searching relevant passages: "`); MiniLM is
   a general-purpose sentence encoder.
2. **The chunker barely matters.** With bge, fixed windows edge out
   heading-sections by 0.023 MRR and tie on recall@5; with MiniLM the order
   flips and sections wins by 0.011 MRR. The "smarter" section chunker - the
   one I expected to win - did not pay for its extra complexity on this
   corpus. `fixed__bge` wins the grid and is the shipped default.
3. **recall@10 = 1.000** for three of the four configs: the right file is
   almost always somewhere in the top ten distinct candidates. The remaining
   errors are ranking errors, not coverage errors.

## Failure analysis

`fixed__bge` puts the truth file at rank 1 for 53/70 questions. The 17
non-rank-1 cases fall into four recurring patterns:

**1. Sibling-page confusion (10 of 17).** The docs explain most topics two or
three times - tutorial, advanced, and API reference. The retriever lands on
the sibling, truth at rank 2. Examples: *"Do my test functions need to be
async def?"* -> `advanced/async-tests.md` beats `tutorial/testing.md` (q026);
*"How can I detect when a client disconnects from a WebSocket?"* ->
`reference/websockets.md` beats `advanced/websockets.md` (q049); *"security
scheme types in OpenAPI"* -> `security/first-steps.md` beats
`security/index.md` (q056). File-level truth punishes these near-hits hard;
an answer-level metric would score most of them as successes, since the
sibling often contains an equivalent answer.

**2. Multi-home answers (3 of 17).** Some questions are genuinely answered in
more than one file, and the golden set's single-truth assumption bites. *"How
do I declare a JSON request body?"* is answered in `tutorial/body.md` (truth)
but also, in passing, in `body-multiple-params.md` (retrieved first, q012).
Same shape for the install command (q014, root `index.md` vs
`tutorial/index.md`) and partial updates (q023).

**3. Premise-mismatch questions (1 of 17).** *"Does setting a default value
on a path parameter make it optional?"* (q004) embeds to vectors near the
*optional query parameter* pages - the question is about path params, but its
surface vocabulary ("default value", "optional") belongs to another topic.
Answer: no - and the page that says so ranks 3rd behind two
optional-parameter pages.

**4. Specifics buried in long listicle pages (3 of 17, the worst ranks).**
*"GZipMiddleware options"* (q048, rank 6): `advanced/middleware.md` mentions
GZipMiddleware in two short paragraphs of a page that lists many middlewares;
no chunk of it is dominated by the term. Same shape for *"what does `main:app`
mean"* (q041, rank 6) and the `fastapi` CLI app-discovery question (q025,
rank 4).

## Answer quality (LLM-as-judge) - approximate by design

`llama3.1:8b` judged its own pipeline's answers against the references:
**77.1% correct (54/70)** on `fixed__bge` with k=5.

Three honesty caveats, in decreasing order of importance:

1. **The judge is noisy.** Manually auditing all 16 "incorrect" verdicts:
   roughly a third are judge mistakes - it misquoted answers that actually
   agree with the reference (q027: the answer correctly says `BackgroundTask`
   comes from `starlette.background`; the judge claims it said the opposite;
   same shape in q019 and q066). Others are mixed cases (q043 names the
   correct 307 default but ships a wrong code example - the judge's verdict
   is right for a partly wrong reason). The rest are real generation
   failures (wrong
   mechanism named: q006, q059; imprecise defaults: q031; incomplete: q053)
   and one honest *"I don't know"* caused by the k=5 retrieval miss on q048.
   True answer quality is plausibly ~85%.
2. **The judge is prompt-sensitive.** Three near-identical judge setups -
   the original strict prompt/parser, a revised prompt that stopped
   penalizing concise answers, and the same prompt after a
   punctuation-only normalization - scored the same pipeline 88.6%, 80.0%,
   and 77.1%. An 11.5-point spread from changes that plausibly should not
   matter: treat any single judged_correct number as a rough estimate, not
   a measurement.
3. **Same-model bias.** Generator and judge are the same model. The retrieval
   metrics above are the load-bearing numbers; the judge score is secondary
   evidence.

## Limitations

- One corpus, one domain (developer docs), English only. Nothing here
  generalizes automatically to other corpora.
- Single ground-truth file per question undercounts sibling-page near-hits
  (pattern 1 above) - a graded relevance scale would be fairer but needs
  per-question relevance judgments across 140 files.
- The golden set was authored from the corpus files directly; questions may
  share vocabulary with their source pages more than real user questions
  would, which flatters retrieval slightly.
- No reranker, no hybrid/BM25 search, no query rewriting - deliberately out
  of scope; the harness exists so that adding any of them becomes a
  measurable experiment instead of a vibe.

## Reproducing

```
rag fetch    # corpus at the pinned tag
rag ingest   # build all four indexes (writes the ingest manifest)
rag eval     # retrieval grid -> data/results/
rag eval --judge   # + answer generation and judging on the default config
```

The eval refuses to run if any index was built from a different corpus state
than the one on disk (ingest manifest check), so the four configs are always
compared on equal footing.
