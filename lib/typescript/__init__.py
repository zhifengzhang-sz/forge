"""TypeScript language module."""

from pathlib import Path

from lib.common.types import TopicConfig, Unit
from lib.typescript.walk import walk_ts_files
from lib.typescript.extract import extract_units_from_file, extract_diffs
from lib.typescript.score import filter_by_quality


class TypeScriptModule:
    def walk(self, repo_path: Path, config: TopicConfig) -> list[dict]:
        return walk_ts_files(repo_path, config)

    def extract(self, file_info: dict, domain: str) -> list[Unit]:
        return extract_units_from_file(file_info, domain)

    def extract_diffs(self, repo_path: Path, domain: str) -> list[Unit]:
        return extract_diffs(repo_path, domain)

    def score(self, units: list[Unit], config: TopicConfig) -> list[Unit]:
        return filter_by_quality(units, config)
