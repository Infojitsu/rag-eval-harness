"""LLM-as-judge answer scoring.

Approximate by design: the judge is the same local model family as the
generator, so scores are secondary evidence next to the retrieval metrics.
The eval report states this bias explicitly.
"""
import re

JUDGE_PROMPT = """You are grading a RAG system's answer against a reference answer.

Question: {question}

Reference answer: {reference}

Generated answer: {answer}

Does the generated answer correctly answer the question? Use the reference
answer as the source of truth. The generated answer does NOT need to include
every detail of the reference - judge only the facts essential to the
question. Penalize contradictions with the reference and missing essential
facts; ignore style, phrasing, and harmless extra detail.

Reply with exactly one word first - CORRECT or INCORRECT - optionally
followed by a short reason."""


def _parse_verdict(raw: str) -> str:
    # Local models wrap the verdict ("`CORRECT`", "**CORRECT**", "Verdict:
    # CORRECT"), so scan the first few words stripped of non-letters.
    for word in raw.strip().split()[:3]:
        token = re.sub(r"[^A-Za-z]", "", word).upper()
        if token == "INCORRECT":
            return "incorrect"
        if token == "CORRECT":
            return "correct"
    # Anything without an unambiguous verdict counts as incorrect.
    return "incorrect"


def judge_answer(question: str, reference: str, answer: str, client) -> dict:
    raw = client.generate(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer),
        temperature=0.0,
    )
    return {"verdict": _parse_verdict(raw), "raw": raw}
