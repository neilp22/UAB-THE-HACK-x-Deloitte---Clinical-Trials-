"""Central configuration loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env — search repo root, then parent directory (for dev setups
# where the .env lives one level above the cloned repo)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")           # repo root first
load_dotenv(_ROOT.parent / ".env")    # parent fallback (never overrides)

# --- LLM ---
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2048"))

# --- ClinicalTrials.gov API ---
CT_API_BASE: str = "https://clinicaltrials.gov/api/v2"
CT_PAGE_SIZE: int = int(os.getenv("CT_PAGE_SIZE", "200"))
CT_MAX_RESULTS: int = 1000   # ceiling for any single query
CT_MAX_RETRIES: int = 3
CT_RETRY_BASE_DELAY: float = 1.0
CT_POLITE_DELAY: float = 0.3  # seconds between paginated requests

# --- Paths ---
DATA_DIR: Path = _ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
TREC_2021_DIR: Path = DATA_DIR / "trec" / "2021"
TREC_2022_DIR: Path = DATA_DIR / "trec" / "2022"
OUTPUT_DIR: Path = _ROOT / "output"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
