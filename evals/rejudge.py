"""Judge the answers of a saved run again, without running the agent.

Run: uv run python -m evals.rejudge runs/<run>.json --label baseline-rejudged
The answers stay exactly the same, so any change in judge_pass comes from the judge alone.
Then compare: uv run python -m evals.compare runs/<run>.json runs/<new run>.json
"""

import argparse
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from evals.compare import load_run
from evals.run import (
    RunReport,
    format_table,
    git_commit,
    judge_trace,
    output_path,
    parse_label,
    save_report,
)
from evals.score import summarize
from policy_assistant.config import ROOT, load_settings


def rejudge_report(
    report: RunReport, client, judge_model: str, label: str, source: str, now: datetime
) -> RunReport:
    """Return a copy of the report with every answer judged again. Nothing else is re-run."""
    results = []
    for i, result in enumerate(report.results, 1):
        print(f"[{i}/{len(report.results)}] {result.case.id} #{result.attempt}", flush=True)
        score = result.score.model_copy()
        score.judge_pass, score.judge_reason = judge_trace(
            result.case, result.trace, client, judge_model
        )
        results.append(result.model_copy(update={"score": score}))

    return report.model_copy(
        update={
            "label": label,
            "created_at": now.isoformat(timespec="seconds"),
            "git_commit": git_commit(),  # the judge's code version; the answers are older
            "judge_model": judge_model,
            "rejudged_from": source,
            "summary": summarize([r.score for r in results]),
            "results": results,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--label", required=True, type=parse_label)
    args = parser.parse_args()

    now = datetime.now()
    path = output_path(args.label, now)
    if path.exists():
        parser.error(f"{path.relative_to(ROOT)} already exists; use another label.")

    settings = load_settings()
    report = rejudge_report(
        load_run(args.run), OpenAI(), settings.judge_model, args.label, args.run.name, now
    )
    save_report(path, report.model_dump_json(indent=2))
    print()
    print(format_table(report.results, report.summary))
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
