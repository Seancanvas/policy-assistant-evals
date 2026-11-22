"""Ask one question from the command line.

Run: uv run python -m policy_assistant.cli "保單 NW-TRV-0001 還有效嗎？"
"""

import argparse

from dotenv import load_dotenv
from openai import OpenAI

from policy_assistant.agent import run_agent
from policy_assistant.config import INDEX_DIR, POLICIES_PATH, load_settings
from policy_assistant.retrieval import Retriever, openai_query_embedder
from policy_assistant.tools import ToolRunner, load_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the policy assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true", help="Print the full trace as JSON.")
    args = parser.parse_args()

    load_dotenv()
    settings = load_settings()
    client = OpenAI()
    retriever = Retriever.load(INDEX_DIR, openai_query_embedder(client, settings.embedding_model))
    runner = ToolRunner(retriever, load_policies(POLICIES_PATH), settings.top_k)

    trace = run_agent(args.question, client, runner, settings.model, settings.max_turns)

    if args.trace:
        print(trace.model_dump_json(indent=2))
        return
    answer = trace.answer
    print(f"[{answer.status}] {answer.answer}")
    if answer.citations:
        print("Citations:", ", ".join(answer.citations))
    print(
        f"({len(trace.tool_calls)} tool calls, {trace.turns} turns, "
        f"{trace.input_tokens}+{trace.output_tokens} tokens, {trace.latency_ms / 1000:.1f}s)"
    )


if __name__ == "__main__":
    main()
