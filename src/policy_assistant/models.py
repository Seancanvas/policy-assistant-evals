from typing import Literal

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    """One clause of a policy document; the unit we embed, retrieve and cite."""

    doc_id: str
    section_id: str  # e.g. "3.1"
    title: str
    text: str
    lang: Literal["zh", "en"]
    version: str

    @property
    def citation_id(self) -> str:
        return f"{self.doc_id}#{self.section_id}"


class Policy(BaseModel):
    policy_id: str
    holder_name: str
    product: str
    status: Literal["active", "expired", "cancelled"]
    start_date: str  # ISO date
    end_date: str  # ISO date
    riders: list[str] = []


class Answer(BaseModel):
    """The agent's final, structured answer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "not_found", "error"]
    answer: str
    citations: list[str]  # e.g. ["travel-insurance#3.1"]


class ToolCallRecord(BaseModel):
    turn: int  # 1-based; calls with the same turn were requested together
    name: str
    arguments: dict
    output: str
    duration_ms: float


class Trace(BaseModel):
    question: str
    tool_calls: list[ToolCallRecord] = []
    answer: Answer | None = None
    turns: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
