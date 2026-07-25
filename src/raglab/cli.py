"""Typer CLI: rag fetch / ingest / ask / eval."""
import typer

from raglab import store
from raglab.chunking import chunk_documents
from raglab.config import CONFIGS, DEFAULT_CONFIG, FASTAPI_TAG, RESULTS_DIR, TOP_K
from raglab.corpus import corpus_files, fetch_corpus, load_documents
from raglab.evalharness.golden import load_golden
from raglab.evalharness.runner import (
    evaluate_answers,
    results_table,
    run_grid,
    save_answers,
    save_results,
)
from raglab.generate import answer
from raglab.retrieval import get_embedder

app = typer.Typer(
    help="RAG over the FastAPI docs, with a real evaluation harness.",
    no_args_is_help=True,
    add_completion=False,
)


def _config(name: str):
    if name not in CONFIGS:
        raise typer.BadParameter(f"Unknown config '{name}'. Choose from: {', '.join(CONFIGS)}")
    return CONFIGS[name]


@app.command()
def fetch():
    """Download the FastAPI docs corpus at the pinned tag."""
    count = fetch_corpus()
    typer.echo(f"Fetched {count} markdown files into data/corpus/")


@app.command()
def ingest(
    config: str = typer.Option(None, help="Config name; default: all four."),
):
    """Chunk, embed, and index the corpus (one Chroma collection per config)."""
    docs = load_documents()
    typer.echo(f"Loaded {len(docs)} documents.")
    client = store.get_client()
    targets = [_config(config)] if config else list(CONFIGS.values())
    for cfg in targets:
        chunks = chunk_documents(docs, cfg.chunker)
        typer.echo(f"[{cfg.name}] {len(chunks)} chunks; embedding with {cfg.embed_model} ...")
        embeddings = get_embedder(cfg.embed_model).encode_passages([c.text for c in chunks])
        store.build_collection(client, cfg.name, chunks, embeddings)
        store.record_ingest(cfg.name, FASTAPI_TAG, len(docs), len(chunks))
        typer.echo(f"[{cfg.name}] indexed.")


@app.command()
def ask(
    question: str,
    config: str = typer.Option(DEFAULT_CONFIG),
    k: int = typer.Option(TOP_K),
):
    """Answer a question from the indexed docs, with cited sources."""
    text, chunks = answer(question, _config(config), k=k)
    typer.echo("\n" + text + "\n")
    typer.echo("Sources:")
    for i, chunk in enumerate(chunks, start=1):
        typer.echo(f"  [{i}] {chunk.source_file} (score {chunk.score:.3f})")


@app.command("eval")
def eval_command(
    judge: bool = typer.Option(False, "--judge/--no-judge", help="Also grade generated answers with the LLM judge."),
    judge_config: str = typer.Option(DEFAULT_CONFIG, help="Config to answer+judge with."),
):
    """Run the golden set through all four configs and report retrieval metrics."""
    examples = load_golden(corpus_files=corpus_files())
    # Refuse stale or partial indexes: every collection must have been built
    # from the corpus state we are evaluating against.
    client = store.get_client()
    n_docs = len(load_documents())
    for cfg in CONFIGS.values():
        store.verify_index(client, cfg.name, FASTAPI_TAG, n_docs)
    typer.echo(f"Evaluating {len(examples)} golden questions across {len(CONFIGS)} configs ...")
    results = run_grid(examples, client=client)
    typer.echo("\n" + results_table(results) + "\n")
    save_results(results)
    typer.echo(f"Saved results and per-config misses to {RESULTS_DIR}")
    if judge:
        cfg = _config(judge_config)
        typer.echo(f"Judging generated answers for {cfg.name} (2 LLM calls per question) ...")
        judged = evaluate_answers(examples, cfg)
        typer.echo(f"LLM-judged correct: {judged['judged_correct']:.1%} of {judged['n']}")
        save_answers(judged)
        typer.echo(f"Saved answer transcripts to {RESULTS_DIR}")


def main():
    app()


if __name__ == "__main__":
    main()
