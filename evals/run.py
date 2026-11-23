"""Run every case and save the results.

Run: uv run python -m evals.run --label baseline
Output: runs/<YYYY-MM-DD_HHMM>_<label>.json with per-case traces, scores and a summary.
"""

import argparse
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from openai import APIError, OpenAI
from pydantic import BaseModel

from evals.judge import judge
from evals.score import (
    Case,
    CaseScore,
    load_cases,
    retrieved_citation_ids,
    score_case,
    summarize,
)
from policy_assistant.agent import run_agent
from policy_assistant.config import INDEX_DIR, POLICIES_PATH, ROOT, RUNS_DIR, load_settings
from policy_assistant.models import Answer, Trace
from policy_assistant.retrieval import Retriever, openai_query_embedder
from policy_assistant.tools import ToolRunner, load_policies


class CaseResult(BaseModel):
    case: Case
    trace: Trace
    retrieved_ids: list[str]
    score: CaseScore


def evaluate_case(
    case: Case, client, runner, model: str, max_turns: int, judge_model: str
) -> CaseResult:
    """Run, score and judge one case. An API error fails this case instead of the whole run."""
    try:
        trace = run_agent(case.question, client, runner, model, max_turns)
    except APIError as exc:
        trace = Trace(
            question=case.question,
            answer=Answer(status="error", answer=f"API error: {exc}", citations=[]),
        )

    retrieved = retrieved_citation_ids(trace)
    score = score_case(case, trace, retrieved)

    answer = trace.answer
    if answer is None or answer.status == "error":
        score.judge_pass, score.judge_reason = False, "No answer to judge."
    else:
        try:
            score.judge_pass, score.judge_reason = judge(
                client, judge_model, case.question, case.reference_answer, answer.answer
            )
        except APIError as exc:
            score.judge_reason = f"Judge API error: {exc}"  # judge_pass stays None

    return CaseResult(case=case, trace=trace, retrieved_ids=retrieved, score=score)


def choose_cases(cases: list[Case], only: list[str] | None) -> list[Case]:
    if not only:
        return cases
    known = {c.id for c in cases}
    unknown = [i for i in only if i not in known]
    if unknown:
        raise ValueError(f"Unknown case IDs: {', '.join(unknown)}")
    return [c for c in cases if c.id in set(only)]


def run_stats(results: list[CaseResult]) -> dict[str, float]:
    traces = [r.trace for r in results]
    latencies = [t.latency_ms for t in traces]
    return {
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_max": max(latencies),
        "turns_mean": statistics.mean(t.turns for t in traces),
        "tool_calls_mean": statistics.mean(len(t.tool_calls) for t in traces),
        "input_tokens_total": sum(t.input_tokens for t in traces),
        "output_tokens_total": sum(t.output_tokens for t in traces),
    }


def git_commit() -> str:
    """Short HEAD SHA, with '-dirty' if there are uncommitted changes, so a run is traceable."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("Y" if value else "N")


# Column header -> metric. Short so one case fits on one line.
COLUMNS = {
    "ret": "retrieval_hit",
    "cite": "citation_ok",
    "grnd": "citations_grounded",
    "tool": "tool_choice_ok",
    "stat": "status_ok",
    "incl": "must_include_ok",
    "excl": "must_not_include_ok",
    "judge": "judge_pass",
}


def format_table(results: list[CaseResult], summary: dict[str, float]) -> str:
    header = f"{'case':<5} {'category':<10} " + " ".join(f"{h:>5}" for h in COLUMNS)
    lines = [header]
    for r in results:
        marks = " ".join(f"{_mark(getattr(r.score, m)):>5}" for m in COLUMNS.values())
        lines.append(f"{r.case.id:<5} {r.case.category:<10} {marks}")
    rates = " ".join(
        f"{summary[m]:>5.0%}" if m in summary else f"{'-':>5}" for m in COLUMNS.values()
    )
    lines.append(f"{'rate':<5} {'':<10} {rates}")
    return "\n".join(lines)


def output_path(label: str, now: datetime) -> Path:
    return RUNS_DIR / f"{now:%Y-%m-%d_%H%M}_{label}.json"


class RunReport(BaseModel):
    label: str
    created_at: str
    git_commit: str
    model: str
    judge_model: str
    embedding_model: str
    top_k: int
    max_turns: int
    summary: dict[str, float]
    stats: dict[str, float]
    results: list[CaseResult]


def _label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("use only letters, digits, '-' and '_'")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, type=_label)
    parser.add_argument("--cases", default="evals/cases.jsonl")
    parser.add_argument("--only", nargs="*", help="Run only these case IDs.")
    args = parser.parse_args()

    settings = load_settings()
    cases = choose_cases(load_cases(Path(args.cases)), args.only)
    client = OpenAI()
    retriever = Retriever.load(INDEX_DIR, openai_query_embedder(client, settings.embedding_model))
    runner = ToolRunner(retriever, load_policies(POLICIES_PATH), settings.top_k)

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id} {case.question[:50]}", flush=True)
        results.append(
            evaluate_case(
                case, client, runner, settings.model, settings.max_turns, settings.judge_model
            )
        )

    now = datetime.now()
    summary = summarize([r.score for r in results])
    report = RunReport(
        label=args.label,
        created_at=now.isoformat(timespec="seconds"),
        git_commit=git_commit(),
        model=settings.model,
        judge_model=settings.judge_model,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        max_turns=settings.max_turns,
        summary=summary,
        stats=run_stats(results),
        results=results,
    )
    path = output_path(args.label, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print()
    print(format_table(results, summary))
    print()
    for key, value in report.stats.items():
        print(f"{key}: {value:,.0f}" if "tokens" in key else f"{key}: {value:,.1f}")
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
