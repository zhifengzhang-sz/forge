"""Generate instruction prompts from code units using the Claude API. Generic across all languages."""

import json
import time
import random
import re
import logging
from pathlib import Path

from lib.common.types import Unit

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a training data generator for a specialised coding model.

Given a code unit and its domain label, generate ONE natural instruction that a developer would give to produce this code.

Rules:
- Name the domain pattern explicitly (e.g. "fp-ts Either", "XState v5 actor", "RxJS observable")
- Be specific enough that a simpler or generic version would NOT be a valid answer
- Phrase as a task: "implement...", "create...", "write..."
- Do NOT use "explain", "describe", "what is", or any non-task phrasing
- Output ONLY the instruction text, nothing else"""

REJECT_PATTERNS = re.compile(r"\b(explain|describe|what is|what are|how does|why)\b", re.IGNORECASE)

MAX_RETRIES = 5
BASE_DELAY = 1.0


def _estimate_cost(units: list[Unit], price_per_mtok_input: float = 3.0, price_per_mtok_output: float = 15.0) -> float:
    total_input_chars = sum(len(u.code) + len(u.imports) + 500 for u in units)
    total_input_tokens = total_input_chars / 4
    total_output_tokens = len(units) * 50
    input_cost = (total_input_tokens / 1_000_000) * price_per_mtok_input
    output_cost = (total_output_tokens / 1_000_000) * price_per_mtok_output
    return input_cost + output_cost


def _call_api(client, code: str, domain: str, imports: str = "") -> str | None:
    user_content = f"Domain: {domain}\n\n"
    if imports:
        user_content += f"Imports:\n{imports}\n\n"
    user_content += f"Code:\n{code}"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "500" in error_str or "502" in error_str or "503" in error_str:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                log.warning("API error (attempt %d/%d): %s. Retrying in %.1fs", attempt + 1, MAX_RETRIES, e, delay)
                time.sleep(delay)
            else:
                log.error("API error (non-retryable): %s", e)
                return None

    log.error("Max retries exceeded")
    return None


def _validate_instruction(instruction: str, domain: str, domain_terms: list[str] | None = None) -> bool:
    """Validate that the generated instruction is usable."""
    if not instruction or len(instruction) < 20:
        return False
    if REJECT_PATTERNS.search(instruction):
        return False
    if domain_terms and not any(t.lower() in instruction.lower() for t in domain_terms):
        return False
    return True


def generate_instructions(
    units: list[Unit],
    output_path: Path,
    metadata_path: Path,
    rejected_path: Path,
    domain_terms: dict[str, list[str]] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Generate instruction-completion pairs from code units.

    Args:
        units: List of scored, deduped, balanced units.
        output_path: Path for training JSONL (messages only).
        metadata_path: Path for sidecar metadata JSONL.
        rejected_path: Path for rejected instructions.
        domain_terms: Optional dict mapping domain name to validation terms.
        dry_run: If True, estimate cost and exit without calling the API.
    """
    import anthropic
    client = anthropic.Anthropic()

    cost = _estimate_cost(units)
    log.info("Estimated API cost: $%.2f for %d units", cost, len(units))

    if dry_run:
        log.info("Dry run -- exiting without API calls")
        return []

    confirm = input(f"Estimated cost: ${cost:.2f} for {len(units)} API calls. Proceed? [y/N] ")
    if confirm.lower() != "y":
        log.info("Aborted by user")
        return []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    domain_terms = domain_terms or {}
    results = []
    rejected_count = 0

    with (
        open(output_path, "w") as f_out,
        open(metadata_path, "w") as f_meta,
        open(rejected_path, "w") as f_rej,
    ):
        for i, unit in enumerate(units):
            if (i + 1) % 100 == 0:
                log.info("Progress: %d/%d (%.0f%%)", i + 1, len(units), (i + 1) / len(units) * 100)

            instruction = _call_api(client, unit.code, unit.domain, unit.imports)
            terms = domain_terms.get(unit.domain)

            if not instruction or not _validate_instruction(instruction, unit.domain, terms):
                rejected_count += 1
                f_rej.write(json.dumps({
                    "source": unit.source,
                    "domain": unit.domain,
                    "instruction": instruction or "",
                    "reason": "validation_failed" if instruction else "api_error",
                }) + "\n")
                continue

            completion = unit.code
            if unit.imports:
                completion = unit.imports + "\n\n" + completion

            example_id = f"{unit.fingerprint or 'unknown'}-{i:04d}"

            f_out.write(json.dumps({
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": completion},
                ],
                "id": example_id,
            }) + "\n")

            f_meta.write(json.dumps({
                "id": example_id,
                "domain": unit.domain,
                "source": unit.source,
                "unit_type": unit.unit_type,
                "quality_score": unit.quality_score,
            }) + "\n")

            results.append({"unit": unit, "instruction": instruction, "id": example_id})

    log.info("Generated %d training examples (%d rejected)", len(results), rejected_count)
    return results
