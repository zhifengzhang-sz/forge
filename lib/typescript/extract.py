"""Extract semantic units from TypeScript files. Brace-matching parser."""

import subprocess
import logging
from pathlib import Path

from lib.common.types import Unit

log = logging.getLogger(__name__)


def _extract_imports(content: str) -> str:
    """Extract the import block from the top of a file."""
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(line)
        elif stripped.startswith("//") or stripped == "":
            continue
        elif lines:
            break
    return "\n".join(lines)


def _extract_exported_declarations(content: str) -> list[tuple[str, str]]:
    """Extract exported declarations with their bodies.

    Returns list of (code, unit_type) tuples.
    """
    units = []
    lines = content.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("export ") and (
            "function " in stripped
            or "const " in stripped
            or "type " in stripped
            or "interface " in stripped
        ):
            # Classify by the export keyword on the first line
            if "type " in stripped or "interface " in stripped:
                unit_type = "type"
            else:
                unit_type = "function"

            # Collect the full declaration including braces
            brace_depth = 0
            started = False
            unit_lines = []

            while i < len(lines):
                unit_lines.append(lines[i])
                for ch in lines[i]:
                    if ch == "{":
                        brace_depth += 1
                        started = True
                    elif ch == "}":
                        brace_depth -= 1

                i += 1

                if started and brace_depth == 0:
                    break
                if not started and lines[i - 1].strip().endswith(";"):
                    break
                if not started and i < len(lines) and not lines[i].strip().startswith("|") and unit_lines[-1].strip().endswith(";"):
                    break

            unit_text = "\n".join(unit_lines).strip()
            if len(unit_text) > 50:
                units.append((unit_text, unit_type))
        else:
            i += 1

    return units


def extract_units_from_file(file_info: dict, domain: str) -> list[Unit]:
    """Extract semantic units from a single .ts file."""
    content = file_info["content"]
    file_path = file_info["path"]
    repo_path = file_info["repo_path"]

    imports = _extract_imports(content)
    rel_path = file_path.relative_to(repo_path)
    source = f"{repo_path.name}:{rel_path}"

    units = []
    for code, unit_type in _extract_exported_declarations(content):
        units.append(Unit(
            code=code,
            imports=imports,
            domain=domain,
            source=source,
            unit_type=unit_type,
        ))

    return units


def extract_diffs(repo_path: Path, domain: str, max_commits: int = 300) -> list[Unit]:
    """Extract TypeScript-only diffs from git history."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-merges", f"-n{max_commits}"],
            cwd=repo_path, capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError:
        log.warning("Cannot extract diffs from %s (shallow clone?)", repo_path.name)
        return []

    units = []
    for line in result.stdout.strip().splitlines():
        commit_hash = line.split()[0]

        try:
            diff_result = subprocess.run(
                ["git", "show", commit_hash, "--", "*.ts"],
                cwd=repo_path, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError:
            continue

        diff_text = diff_result.stdout.strip()
        if len(diff_text) < 100 or len(diff_text) > 10000:
            continue

        commit_msg = line[len(commit_hash):].strip()
        units.append(Unit(
            code=diff_text,
            imports="",
            domain=domain,
            source=f"{repo_path.name}@{commit_hash}",
            unit_type="diff",
            commit_message=commit_msg,
        ))

    log.info("Extracted %d diffs from %s", len(units), repo_path.name)
    return units
