"""Load the project's .env for scripts that need the LLM settings. Never prints or returns values."""
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ENV = Path(__file__).resolve().parents[1] / ".env"


def load_project_env(path: Path | None = None) -> bool:
    """Set variables from .env that are not already in the environment; True if a file was found."""
    return bool(load_dotenv(path or PROJECT_ENV, override=False))
