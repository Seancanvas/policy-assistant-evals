"""Ask one question from the command line.

Run: uv run python -m policy_assistant.cli "保單 NW-TRV-0001 還有效嗎？"
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the policy assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true", help="Print the full trace as JSON.")
    parser.parse_args()
    raise NotImplementedError(
        "TODO day 2: load settings, Retriever.load, load_policies, run_agent, print the answer"
    )


if __name__ == "__main__":
    main()
