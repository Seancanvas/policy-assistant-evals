import pytest

SAMPLE_DOC = """\
---
doc_id: sample-doc
title: Sample
lang: en
version: 2026-01
---

# Sample policy

Intro text that is not a clause.

## 1.1 Waiting period
Claims within 30 days of the start date are not covered.

## 1.2 Claim deadline
Claims must be filed within 2 years.
"""


@pytest.fixture
def sample_doc() -> str:
    return SAMPLE_DOC
