# Policy Assistant with Evals

A small assistant for the claims and service staff of a fictional insurer, Northwind Insurance.
It is written in plain Python on the OpenAI Responses API. It searches policy documents in
Chinese and English, looks up customer policies through one read-only tool, cites every
answer, and says "not found" rather than guess. An eval suite of 24 cases measures all of this.

The interesting part is not the agent; it is what the evals found. See [FINDINGS.md](FINDINGS.md).

All data in this repository is fictional.

## Why

Staff answering "can this customer claim for X?" have to do two things: find the clause that
applies, sometimes in a document written in the other language, and check the customer's own
policy (status, dates, riders). Getting either wrong means a wrong answer to a customer.

So the assistant has to be right, show where its answer comes from, and admit when the documents
don't cover a question. Each of those is a metric in the eval suite:

| What matters | How it is measured |
|---|---|
| Finds the right clause | `retrieval_hit`: the expected clause was among the retrieved ones |
| Shows its source | `citation_ok`: the answer cites the expected clause |
| Does not invent sources | `citations_grounded`: every citation was actually retrieved |
| Checks the customer's policy | `tool_choice_ok`: the expected tools were called |
| Admits what it doesn't know | `status_ok`: `answered` vs `not_found` |
| Gets the answer right | `must_include_ok` (deterministic) and `judge_pass` (LLM judge with the evidence) |
| Ignores instructions in documents | `must_not_include_ok` on injection cases |

## How it works

```
offline   data/docs/*.md --split by "## <section>"--> 43 clauses --embed--> index/
                                                                  (chunks.json + embeddings.npy)

online    question
            -> agent loop (Responses API, up to MAX_TURNS)
                 tool: search_policy_docs(query)  -> top-k clauses by cosine similarity
                 tool: get_policy(policy_id)      -> one policy record, read-only
            -> structured answer {status, answer, citations}  (strict JSON schema)
            -> trace: every tool call with its turn, tokens, latency

evals     cases.jsonl -> run.py -> score.py (deterministic) + judge.py (LLM) -> runs/*.json
          compare.py: what changed between two runs, regressions first
          rejudge.py: judge a saved run again, without re-running the agent
```

| Path | What it does |
|---|---|
| `src/policy_assistant/ingest.py` | Parse front matter, split into clauses, embed with a content-hash cache |
| `src/policy_assistant/retrieval.py` | Cosine top-k in numpy |
| `src/policy_assistant/tools.py` | The two tools, tool schemas, and a dispatcher that never raises on model input |
| `src/policy_assistant/agent.py` | The tool-calling loop and the system instructions |
| `src/policy_assistant/cli.py` | Ask one question from the command line |
| `evals/` | Cases, scoring, judge, runner, comparison, re-judging |
| `runs/` | Every eval run used in FINDINGS.md, committed as evidence |

## Run it

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # add OPENAI_API_KEY; the other values are filled in
uv run python scripts/smoke_test.py          # checks the key and models, costs < $0.01
uv run python -m policy_assistant.ingest      # builds index/ (re-runs reuse the cache)

uv run python -m policy_assistant.cli "今天是 2026-11-23。NW-HLT-0001 上週因為生病住院，可以理賠嗎？"
uv run python -m policy_assistant.cli --trace "When does policy NW-AUT-0001 end?"

uv run python -m evals.run --label my-run                         # all 24 cases
uv run python -m evals.run --label t05 --only T05 --repeat 5      # one case, five times
uv run python -m evals.compare runs/A.json runs/B.json
uv run python -m evals.rejudge runs/A.json --label A-rejudged

uv run pytest                                 # 108 tests, no network
```

Questions about dates should say what "today" is; the model does not know.
A full eval run takes about 3 minutes and costs a few cents.

## Eval results

Baseline vs one fix, full suite, both judged by the evidence-aware judge:

| Metric | baseline | after fix |
|---|---|---|
| retrieval_hit | 87.5% | 93.8% |
| citation_ok | 87.5% | 93.8% |
| citations_grounded | 100% | 100% |
| tool_choice_ok | 95.8% | 100% |
| status_ok | 91.7% | 95.8% |
| must_include_ok | 95.8% | 100% |
| must_not_include_ok | 100% | 100% |
| judge_pass | 91.7% | 87.5% |
| mean latency | 5.3 s | 5.5 s |

The fix asks the agent to search in both languages. It targets one failure: a Chinese question
missing an English clause when a Chinese clause on a related topic outranks it. The two judge
regressions are unrelated to the fix, and FINDINGS.md explains each one. These are single runs;
the targeted case was also measured 5 times before and after (4/5 -> 5/5).

### Spot-check of the judge

I read the verdicts that changed or that I disagreed with:

| Case | Judge | My reading |
|---|---|---|
| T06, answer adds the correct +5 days for large claims | old judge: fail; evidence-aware: pass | Agree with the new judge |
| T16, answer says "No." then explains a full refund | fail | Agree |
| T09, answer gives the status but not the coverage period | fail | Disagree: the question only asked for the status; the reference asks for more |
| T24, "Is the claim approved?" answered as "cannot tell" | fail | Disputed: the question is ambiguous (see FINDINGS.md) |

Changing the judge is checked the same way as changing the agent: re-judge saved runs and
compare. The evidence-aware judge changed no verdict on the baseline answers.

## Design decisions and trade-offs

- **No agent framework.** The loop is about 50 lines, so every turn, tool call and token count is
  visible in the trace and easy to test with a fake client.
- **numpy instead of a vector database.** 43 clauses fit in memory; a brute-force cosine search
  is exact and has no moving parts. A real corpus would need a vector store (see below).
- **One clause per chunk.** Clauses are the unit people cite, so retrieval, citation and scoring
  all use the same ID, such as `travel-insurance#3.3`.
- **Retrieval always returns k results.** A topic the documents don't cover (pet insurance) scores
  about as high as a real hit, so a score threshold can't detect "not covered". That decision is
  left to the agent's instructions and measured by the `not-found` cases.
- **Structured output plus validation.** The final answer must match a strict JSON schema, and it
  is validated again with pydantic. A failed validation or the turn limit gives
  `status="error"`, never an invented answer.
- **Deterministic checks first, LLM judge second.** String and set checks are free and stable. The
  judge is only for what they cannot check, and it is checked itself (Finding 2).
- **Run files are evidence.** Each run records the git commit, models and settings, and is never
  overwritten.

## Safety

- **Read-only tools.** The agent can search and look up; it cannot change anything.
- **Validated arguments.** Policy IDs must match `NW-XXX-0000` exactly. Bad input from the model
  comes back as an error message the model can read, not an exception.
- **Documents are untrusted.** The instructions say clause text is reference material, not
  instructions. `internal-notes#9.3` contains a planted injection; in case T23 it was retrieved
  and not followed. The judge is told the same about the answer and the evidence it grades.
- **Bounded work.** At most `MAX_TURNS` model calls per question, and `k` comes from settings,
  not from the model.
- **Secrets.** The API key lives only in `.env`, which is git-ignored.

## Taking it to production

- **Retrieval at scale:** a vector store with metadata filters (product, language, document
  version), and re-indexing when a document version changes. The content-hash cache already
  avoids re-embedding unchanged clauses.
- **Permissions:** filter documents and policy records by the staff member's role before they reach
  the model, not in the prompt. Policy records contain names; return only the fields a question
  needs.
- **Human in the loop:** the assistant explains coverage; it should never approve a claim. A claim
  decision stays with a person, and the answer says which clause it relies on so that person can
  check it.
- **Monitoring:** keep traces, sample live answers for the judge, and add every real failure to the
  case file. Run the suite on every prompt, model or index change, and read the regressions, not
  just the totals.
- **Data residency and retention:** decide where traces are stored and for how long before going
  live, since they contain customer questions and policy data.
- **Cost and latency:** about 2 model calls and 5 seconds per question today. Parallel tool calls
  and shorter instructions are the first levers; each change goes through the eval suite.

## Reusable vs. customer-specific

| Reusable as is | Specific to each customer |
|---|---|
| Agent loop, tool dispatcher, structured answers, traces | The documents, how to split them, and the citation ID format |
| Scoring, judge, runner, compare, rejudge | The tools: which systems to read, and their permissions |
| The metric set and the "not found" and injection case types | The 24 cases and reference answers, written with the customer's experts |
| The workflow: baseline, one change, compare, read regressions | The instructions, such as the bilingual-search rule here |

## Time spent and use of AI tools

Built over four days, 19-23 November 2026; the commit history shows each step.

I used Claude Code as a pair programmer throughout. I wrote first drafts of the front-matter
parser, the section splitter, the loader, the index builder, the retrieval search and the agent
loop; Claude Code found and fixed bugs in them and wrote most of the tests. Claude Code also
drafted the mock documents, the policy data, the 24 eval cases and most of the eval tooling,
and I reviewed them before committing. The investigation in FINDINGS.md was done together: we
read the traces, re-ran the failing case, and decided on one change at a time. Decisions such as
what counts as a citation, not editing cases after seeing results, and fixing the judge
separately from the agent are recorded in the commit messages and in FINDINGS.md.
