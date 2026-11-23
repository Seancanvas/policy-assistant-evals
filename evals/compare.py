"""Compare two eval runs.

Run: uv run python -m evals.compare runs/A.json runs/B.json
Prints what changed in the setup, the change per metric, then the cases that got worse and
the cases that got better. Worse comes first: a fix that breaks other cases is not a fix.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from evals.run import RunReport
from evals.score import METRICS

# Setup fields worth seeing side by side. A change here explains a change in the scores.
CONFIG_FIELDS = [
    "git_commit",
    "model",
    "judge_model",
    "embedding_model",
    "top_k",
    "max_turns",
    "repeat",
    "rejudged_from",
]


@dataclass(frozen=True)
class Tally:
    passed: int
    total: int  # runs where the metric applied (None values are left out)

    @property
    def rate(self) -> float:
        return self.passed / self.total

    def __str__(self) -> str:
        return f"{self.passed}/{self.total}"


@dataclass(frozen=True)
class CaseChange:
    case_id: str
    metric: str
    before: Tally
    after: Tally
    reason: str  # judge reason from a failing run after the change, if any


def load_run(path: Path) -> RunReport:
    return RunReport.model_validate_json(path.read_text(encoding="utf-8"))


def case_tallies(report: RunReport) -> dict[str, dict[str, Tally]]:
    """{case_id: {metric: Tally}}, adding up repeated runs of the same case."""
    counts: dict[str, dict[str, list[int]]] = {}
    for result in report.results:
        per_metric = counts.setdefault(result.case.id, {})
        for metric in METRICS:
            value = getattr(result.score, metric)
            if value is None:
                continue
            passed_total = per_metric.setdefault(metric, [0, 0])
            passed_total[0] += value
            passed_total[1] += 1
    return {
        case_id: {m: Tally(p, t) for m, (p, t) in per_metric.items()}
        for case_id, per_metric in counts.items()
    }


def _failing_judge_reason(report: RunReport, case_id: str) -> str:
    for result in report.results:
        if result.case.id == case_id and result.score.judge_pass is False:
            return result.score.judge_reason
    return ""


def case_changes(before: RunReport, after: RunReport) -> list[CaseChange]:
    """Every (case, metric) whose pass rate moved, for cases present in both runs."""
    a, b = case_tallies(before), case_tallies(after)
    changes = []
    for case_id in a.keys() & b.keys():
        for metric in METRICS:
            old, new = a[case_id].get(metric), b[case_id].get(metric)
            if old is None or new is None or old.rate == new.rate:
                continue
            reason = _failing_judge_reason(after, case_id) if metric == "judge_pass" else ""
            changes.append(CaseChange(case_id, metric, old, new, reason))
    return sorted(changes, key=lambda c: (c.case_id, METRICS.index(c.metric)))


def config_changes(before: RunReport, after: RunReport) -> list[tuple[str, object, object]]:
    return [
        (field, getattr(before, field), getattr(after, field))
        for field in CONFIG_FIELDS
        if getattr(before, field) != getattr(after, field)
    ]


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def format_comparison(before: RunReport, after: RunReport) -> str:
    lines = ["Setup changes"]
    changed = config_changes(before, after)
    lines += [f"  {f:<16} {old} -> {new}" for f, old, new in changed] or ["  (none)"]

    lines += ["", f"  {'metric':<20} {'before':>7} {'after':>7} {'change':>8}"]
    for metric in METRICS:
        old, new = before.summary.get(metric), after.summary.get(metric)
        if old is None and new is None:
            continue
        delta = "" if old is None or new is None else f"{(new - old) * 100:+.1f}pp"
        lines.append(f"  {metric:<20} {_pct(old):>7} {_pct(new):>7} {delta:>8}")

    changes = case_changes(before, after)
    worse = [c for c in changes if c.after.rate < c.before.rate]
    better = [c for c in changes if c.after.rate > c.before.rate]
    for title, group in (("Worse (pass -> fail)", worse), ("Better (fail -> pass)", better)):
        lines += ["", f"{title}: {len(group)}"]
        for c in group:
            line = f"  {c.case_id} {c.metric:<20} {c.before} -> {c.after}"
            lines.append(f"{line}  {c.reason}" if c.reason else line)

    a_ids = {r.case.id for r in before.results}
    b_ids = {r.case.id for r in after.results}
    for title, ids in (("Only in before", a_ids - b_ids), ("Only in after", b_ids - a_ids)):
        if ids:
            lines += ["", f"{title}: {', '.join(sorted(ids))}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # judge reasons may be Chinese
    print(format_comparison(load_run(args.before), load_run(args.after)))


if __name__ == "__main__":
    main()
