"""Check that the API key and model work. Costs a fraction of a cent.

Run: uv run python scripts/smoke_test.py
"""

from openai import OpenAI

from policy_assistant.config import load_settings


def main() -> None:
    settings = load_settings()
    client = OpenAI()

    response = client.responses.create(
        model=settings.model,
        input="Reply with exactly: ok",
        max_output_tokens=16,
    )
    print(f"model:      {settings.model} -> {response.output_text!r}")
    print(f"tokens:     {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    embedding = client.embeddings.create(model=settings.embedding_model, input="測試 test")
    print(f"embedding:  {settings.embedding_model} -> {len(embedding.data[0].embedding)} dims")


if __name__ == "__main__":
    main()
