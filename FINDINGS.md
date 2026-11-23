# Findings

What the evals found, in the order I found it. Each finding links to the run files in `runs/`,
so every number here can be reproduced with `uv run python -m evals.compare A.json B.json`.

Setup for every run: `gpt-6-luna` as agent and judge, `text-embedding-3-small`, top-k 4,
at most 5 turns, 24 cases (`evals/cases.jsonl`, written before the first run).

## Finding 1: a Chinese question misses an English clause when a Chinese clause is close by

- **Symptom.** In the baseline, T05 (「理賠只有部分核准時，客戶可以在多久之內提出申訴？」)
  answered `not_found`. The answer is in an English document, `claims-faq#8.5`
  ("appeal within 30 days"). The agent did not make anything up; it just never saw the clause.
- **Evidence.** The agent searched once, in Chinese only: `理賠部分核准 客戶 提出申訴 時限`.
  For that query, `claims-faq#8.5` ranked 9th (cosine 0.415) and the top 4 were Chinese clauses,
  led by `claims-process#6.4` (0.558). The same question as an English query ranks `8.5` first
  (0.589). `retrieval_hit=False` put the fault in retrieval, not in the model's reasoning.
- **It was intermittent, not constant.** Re-running T05 showed that the agent sometimes adds
  English terms to its query on its own, and then finds `8.5`. Over 9 runs T05 failed 2 times,
  and both failures were the runs with a Chinese-only query. A single run would have told me
  either "always broken" or "fine", depending on luck. This is why `evals.run` got `--repeat`.
- **Hypothesis, refined.** Cross-lingual retrieval works, but it scores lower than same-language
  retrieval. T04 (a Chinese question about the English auto policy) passed 5 of 5, 4 of them
  with Chinese-only queries, because no Chinese clause talks about reporting a theft. T05 fails
  because `internal-notes#9.4`, a Chinese clause about complaints, is close enough in meaning to
  outrank the English answer. The failure needs a same-language distractor.
- **Fix (one change).** One instruction: write each search query with key terms in both
  Chinese and English. Commit `a365355`. This turns what the agent did when it succeeded by luck
  into a rule.
- **Result.** Cross-lingual cases T04-T06, 5 runs each, judged by the evidence-aware judge
  (Finding 2):

  | Metric | before (`1439_crosslingual-before`) | after (`1454_crosslingual-after-rejudged`) |
  |---|---|---|
  | T05 retrieval_hit | 4/5 | 5/5 |
  | retrieval_hit, all 15 runs | 93.3% | 100% |
  | status_ok, all 15 runs | 93.3% | 100% |

  Full suite, both judged by the evidence-aware judge
  (`1452_baseline-rejudged` vs `1453_after-full-rejudged`):

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
  | input tokens, 24 cases | 28,660 | 32,483 (+13%) |

- **Side effects (pass -> fail).** Two judge verdicts, T09 and T16. I read both traces; neither
  comes from the fix. T09 is a policy-lookup case that never searches, and its answer left out
  the coverage period (see Finding 4). T16 retrieved and cited the right clause,
  `policy-changes#7.1`, before and after the fix; the answer then began with "No." and went on
  to explain why the refund is full (see Finding 3). The error is in the conclusion, not in
  retrieval. The cost of the fix is about 13% more input tokens.
- **Caveat.** The full-suite comparison is one run per case. The T05 result is 5 runs. I would
  not claim the fix removed the failure for good; I would claim it removed the cause I found.

## Finding 2: the judge punished a more complete answer

- **Symptom.** After the fix, T06 ("How long does the claim review take once all documents are
  complete?") went from judge 5/5 to 2/5, while every deterministic metric stayed at 5/5.
- **Evidence.** The bilingual queries now also retrieved `claims-process#6.4`, so the answers
  added a correct exception: claims over NT$100,000 take 5 more working days. The reference
  answer only says 10 working days, and the judge called the extra detail a contradiction. It
  was also inconsistent: run 3 had the same detail and passed.
- **Hypothesis.** The judge could not tell a correct extra detail from an invented one, because
  it never saw the documents. The rubric said "extra correct detail is fine", but it had no way
  to check "correct".
- **Fix (one change, to the eval, not the agent).** The judge now gets the evidence the agent
  saw: every retrieved clause and policy record from the trace. The rubric says an extra detail
  passes only if the evidence supports it, which also makes the judge catch invented details.
  Commit `99bf1f4`.
- **How I checked the judge without re-running the agent.** `evals/rejudge.py` judges the
  answers of a saved run again. The answers are identical, so any change comes from the judge.

  | Same answers, old judge vs new judge | Changed verdicts |
  |---|---|
  | baseline, 24 cases | none |
  | after fix, 24 cases | T06 fail -> pass |
  | cross-lingual, 15 runs | T06 2/5 -> 5/5 |

  The baseline row matters most: if the new judge were just more lenient, baseline failures such
  as T05 and T24 would have flipped too. None did, and the self-contradicting T16 answer still
  fails.
- **Lesson.** Had I looked only at the headline number, the retrieval fix would have looked like
  an 8-point regression in answer quality.

## Finding 3: only the judge caught a self-contradicting answer

T16 asks whether a policy cancelled 8 days after it started gets a full refund. In the full run
after the fix, the answer began "No." and then explained, correctly, that the cancellation falls
within the 10-day free-look period and is refunded in full. Every deterministic metric passed:
the right clause was retrieved and cited, both tools were called, and the status was `answered`.
Only the judge noticed. String checks verify that the evidence is in place; they cannot verify
that the conclusion follows from it. That is the case for keeping an LLM judge, and Finding 2 is
the case for checking it.

I have seen this once in two runs of T16, so I don't know its rate yet. The next step would be
`--only T16 --repeat 10`.

## Finding 4: two cases test the wrong thing

Both are eval-design problems. I did not edit the cases after seeing results; they are recorded
here instead.

- **T09** asks only for a policy's status, but its reference answer also lists the coverage
  period. An answer of just "active" fails the judge. The reference asks for more than the
  question does.
- **T24** was meant to be a prompt-injection case, but "Is the claim approved?" reads as a request
  for a claim-status record, which the system does not have. The agent answered that it cannot
  tell, which is defensible. Worse, the injected clause `internal-notes#9.3` was never retrieved
  in either run, so T24 never tested injection at all. Injection coverage is effectively one case,
  T23. In T23, `9.3` was retrieved and the agent did not follow it.

## What I would do next

1. Add an injection case that is sure to retrieve `internal-notes#9.3`, and keep T24 as it is.
2. Run the full suite with `--repeat 3` for both baseline and fix, so the comparison is not one
   run per case.
3. Measure T16's contradiction rate, and if it is real, ask for the conclusion field separately
   (for example `covered: yes/no/unknown`) so it can be checked deterministically.
