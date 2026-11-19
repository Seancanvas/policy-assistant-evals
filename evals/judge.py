"""LLM-as-judge: does the answer agree with the reference answer?

Keep the rubric fixed, ask for a structured {"pass": bool, "reason": str}, and spot-check a
few verdicts by hand. Write in the README whether the judge agreed with you.
"""

JUDGE_INSTRUCTIONS = """\
You grade answers from an insurance policy assistant.
Compare the ANSWER with the REFERENCE. Pass only if the answer states the same facts and
conclusion as the reference. Extra correct detail is fine. A missing or contradicting fact fails.
Ignore wording and language differences.
"""


def judge(client, model: str, question: str, reference: str, answer: str) -> tuple[bool, str]:
    raise NotImplementedError("TODO day 2")
