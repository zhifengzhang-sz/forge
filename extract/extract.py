"""Extract semantic units from TypeScript files.

Three unit types:
  1. Exported functions and arrow functions
  2. Type aliases and interfaces
  3. Git commit diffs (extracted separately)
"""

import re
import subprocess
import logging
from pathlib import Path

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


def _extract_exported_functions(content: str) -> list[str]:
    """Extract exported function/const declarations with their bodies."""
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
            # Collect the full declaration including braces
            brace_depth = 0
            paren_depth = 0
            unit_lines = []
            started = False

            while i < len(lines):
                unit_lines.append(lines[i])
                for ch in lines[i]:
                    if ch == "{":
                        brace_depth += 1
                        started = True
                    elif ch == "}":
                        brace_depth -= 1
                    elif ch == "(":
                        paren_depth += 1
                    elif ch == ")":
                        paren_depth -= 1

                i += 1

                # End conditions
                if started and brace_depth == 0:
                    break
                # Single-line type alias or interface without body
                if not started and lines[i - 1].strip().endswith(";"):
                    break
                # Multi-line type with pipe unions (no braces)
                if not started and i < len(lines) and not lines[i].strip().startswith("|") and unit_lines[-1].strip().endswith(";"):
                    break

            unit_text = "\n".join(unit_lines).strip()
            if len(unit_text) > 50:  # skip trivial re-exports
                units.append(unit_text)
        else:
            i += 1

    return units


def extract_units_from_file(file_info: dict) -> list[dict]:
    """Extract semantic units from a single .ts file.

    Args:
        file_info: dict with keys: path, content, domain, repo_path

    Returns:
        List of dicts with keys: code, imports, domain, source, unit_type
    """
    content = file_info["content"]
    domain = file_info["domain"]
    file_path = file_info["path"]
    repo_path = file_info["repo_path"]

    imports = _extract_imports(content)
    rel_path = file_path.relative_to(repo_path)
    source = f"{repo_path.name}:{rel_path}"

    units = []
    for code in _extract_exported_functions(content):
        # Classify by the export keyword on the first line
        first_line = code.split("\n", 1)[0].strip()
        if "type " in first_line or "interface " in first_line:
            unit_type = "type"
        else:
            unit_type = "function"

        units.append({
            "code": code,
            "imports": imports,
            "domain": domain,
            "source": source,
            "unit_type": unit_type,
        })

    return units


def extract_diffs(repo_path: Path, domain: str, max_commits: int = 300) -> list[dict]:
    """Extract TypeScript-only diffs from git history.

    Requires full clone (not --depth=1).
    """
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
        units.append({
            "code": diff_text,
            "imports": "",
            "domain": domain,
            "source": f"{repo_path.name}@{commit_hash}",
            "unit_type": "diff",
            "commit_message": commit_msg,
        })

    log.info("Extracted %d diffs from %s", len(units), repo_path.name)
    return units
