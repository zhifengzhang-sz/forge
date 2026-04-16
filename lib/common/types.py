"""Shared types for the extraction pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class RepoConfig:
    url: str
    name: str


@dataclass
class TopicConfig:
    """Configuration for a single topic (e.g. typescript.fp, typescript.reactive)."""
    name: str                                   # "typescript.fp"
    language: str                                # "typescript"
    repos: list[RepoConfig]
    file_extensions: list[str]                   # [".ts"]
    skip_dirs: list[str]                         # ["node_modules", "dist", ...]
    skip_suffixes: list[str]                     # [".d.ts", ".spec.ts", ...]
    focus_terms: list[str]                       # ["pipe(", "Either<", ...]
    scoring_signals: list[tuple[str, float]]     # [(regex, weight), ...]
    scoring_penalties: list[tuple[str, float]]   # [(regex, penalty), ...]
    min_unit_length: int = 80
    system_prompt_fragment: str = ""             # domain-specific part for Ollama


@dataclass
class Unit:
    """A single extracted semantic unit."""
    code: str
    imports: str
    domain: str                                  # topic name
    source: str                                  # "repo:path" or "repo@hash"
    unit_type: str                               # "function", "type", "diff"
    quality_score: float = 0.0
    fingerprint: str = ""
    commit_message: str = ""


class LanguageModule(Protocol):
    """Protocol for language-specific extraction modules.

    Each language (typescript, haskell, python, etc.) implements this
    protocol. The orchestrator dispatches to the correct module based
    on TopicConfig.language.
    """

    def walk(self, repo_path: Path, config: TopicConfig) -> list[dict]:
        """Walk source files matching the topic's focus terms."""
        ...

    def extract(self, file_info: dict, domain: str) -> list[Unit]:
        """Extract semantic units from a single source file."""
        ...

    def extract_diffs(self, repo_path: Path, domain: str) -> list[Unit]:
        """Extract diffs from git history."""
        ...

    def score(self, units: list[Unit], config: TopicConfig) -> list[Unit]:
        """Score and filter units by quality."""
        ...
