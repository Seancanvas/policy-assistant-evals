import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
POLICIES_PATH = DATA_DIR / "policies.json"
INDEX_DIR = ROOT / "index"
RUNS_DIR = ROOT / "runs"


@dataclass(frozen=True)
class Settings:
    model: str
    embedding_model: str
    judge_model: str
    top_k: int
    max_turns: int


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    model = os.environ.get("MODEL", "")
    if not model:
        raise RuntimeError("MODEL is not set. Copy .env.example to .env and fill it in.")
    return Settings(
        model=model,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        judge_model=os.environ.get("JUDGE_MODEL") or model,
        top_k=int(os.environ.get("TOP_K", "4")),
        max_turns=int(os.environ.get("MAX_TURNS", "5")),
    )
