"""Compare two eval runs.

Run: uv run python -m evals.compare runs/A.json runs/B.json
Prints the change per metric, then the cases that went fail -> pass and pass -> fail.
Always show pass -> fail: a fix that breaks other cases is not a fix.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.parse_args()
    raise NotImplementedError("TODO day 3")


if __name__ == "__main__":
    main()
