# Policy Assistant with Evals

A small assistant for (fictional) Northwind Insurance staff, built in Python on the OpenAI
Responses API. It searches policy documents, looks up policies with one read-only tool, cites
every answer, and is measured with an eval suite.

> Status: work in progress.

## Why
<!-- Users, their pain point today, and how success is measured. -->

## Run it
```bash
uv sync
cp .env.example .env   # add your key and model
uv run python -m policy_assistant.ingest
uv run python -m policy_assistant.cli "保單 NW-TRV-0001 還有效嗎？"
uv run python -m evals.run --label baseline
uv run pytest
```

## Design decisions and trade-offs
<!-- No framework, read-only tool, numpy instead of a vector database, ... -->

## Eval results
<!-- Baseline vs fix table. Details in FINDINGS.md. -->

## Safety
<!-- Read-only tool, argument validation, documents treated as untrusted, turn limit. -->

## Taking it to production
<!-- Vector store, permission filtering, monitoring, data residency, human review. -->

## Reusable vs. customer-specific
<!-- What the next customer could reuse, and what would be custom. -->

## Time spent and use of AI tools
<!-- Honest account. -->

All data in this repository is fictional.
