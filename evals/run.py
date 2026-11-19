"""Run every case and save the results.

Run: uv run python -m evals.run --label baseline
Output: runs/<YYYY-MM-DD_HHMM>_<label>.json with per-case traces, scores and a summary.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cases", default="evals/cases.jsonl")
    parser.add_argument("--only", nargs="*", help="Run only these case IDs.")
    parser.parse_args()
    raise NotImplementedError(
        "TODO day 2: load cases, run_agent for each, score_case + judge, summarize, save JSON, "
        "print the summary table"
    )


if __name__ == "__main__":
    main()
