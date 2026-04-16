"""LLM-as-judge for training data quality verification.

Uses Claude to evaluate whether extracted code units are good training
examples for domain-specific fine-tuning. This is a semantic quality
filter that runs after regex-based scoring.

The judge evaluates each unit on:
  1. Pattern clarity — does it demonstrate a clear, learnable pattern?
  2. Idiomaticity — is this how an expert would write it?
  3. Self-containedness — can the model learn from this without external context?
  4. Domain signal — does it teach something specific to the target domain?
  5. Training value — would including this improve model output?
"""

import json
import time
import random
import logging
from pathlib import Path

from lib.common.types import Unit

log = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a training data quality judge for a code fine-tuning pipeline.

You evaluate whether a code unit is a GOOD training example for teaching a language model to write idiomatic code in a specific domain.

Rate the unit 1-5:
  5 = Excellent. Clear pattern, idiomatic, self-contained, high domain signal. A model trained on this would produce better code.
  4 = Good. Solid example with minor issues (slightly verbose, one unclear part). Worth including.
  3 = Borderline. Has some value but also noise (boilerplate wiring, trivial re-export, incomplete pattern). Include only if dataset is small.
  2 = Poor. Mostly boilerplate, configuration, or glue code with little domain signal. Skip.
  1 = Bad. Wrong patterns, anti-patterns, or no domain relevance. Definitely skip.

Respond with ONLY a JSON object:
{"score": N, "reason": "one sentence explaining your rating"}"""

MAX_RETRIES = 3
BASE_DELAY = 1.0


def _estimate_cost(units: list[Unit], price_per_mtok_input: float = 0.80, price_per_mtok_output: float = 4.0) -> float:
    """Estimate cost using Haiku pricing."""
    total_input_chars = sum(len(u.code) + len(u.imports) + 800 for u in units)
    total_input_tokens = total_input_chars / 4
    total_output_tokens = len(units) * 30
    input_cost = (total_input_tokens / 1_000_000) * price_per_mtok_input
    output_cost = (total_output_tokens / 1_000_000) * price_per_mtok_output
    return input_cost + output_cost


def _judge_unit(client, unit: Unit, model: str) -> dict | None:
    """Judge a single unit. Returns {"score": int, "reason": str} or None on failure."""
    user_content = f"Domain: {unit.domain}\nUnit type: {unit.unit_type}\n\n"
    if unit.imports:
        user_content += f"Imports:\n{unit.imports}\n\n"
    user_content += f"Code:\n{unit.code}"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=100,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            text = response.content[0].text.strip()
            result = json.loads(text)
            if "score" in result and isinstance(result["score"], int):
                return result
            log.warning("Judge returned invalid format: %s", text)
            return None
        except json.JSONDecodeError:
            log.warning("Judge returned non-JSON: %s", response.content[0].text[:100])
            return None
        except Exception as e:
            status = getattr(e, "status_code", None)
            retryable = (status is not None and (status == 429 or 500 <= status < 600))
            if retryable and attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                log.warning("Judge API error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e)
                time.sleep(delay)
            else:
                log.error("Judge API error: %s", e)
                return None

    return None


def judge_units(
    units: list[Unit],
    output_path: Path | None = None,
    model: str = "claude-haiku-4-5-20251001",
    min_score: int = 3,
    dry_run: bool = False,
) -> list[Unit]:
    """Run LLM judge on units and filter by score.

    Args:
        units: Units to judge (typically after regex scoring).
        output_path: Optional path to write judge results JSONL.
        model: Claude model to use (default: Haiku for cost).
        min_score: Minimum score to keep (default: 3).
        dry_run: Estimate cost without calling API.

    Returns:
        Units that passed the judge (score >= min_score).
    """
    import anthropic
    client = anthropic.Anthropic()

    cost = _estimate_cost(units)
    log.info("Judge: %d units, estimated cost $%.2f (model: %s)", len(units), cost, model)

    if dry_run:
        log.info("Dry run — skipping judge")
        return units

    confirm = input(f"Run LLM judge on {len(units)} units? Estimated cost: ${cost:.2f}. Proceed? [y/N] ")
    if confirm.lower() != "y":
        log.info("Judge aborted by user")
        return units

    passed = []
    rejected = 0
    errors = 0
    score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    f_out = open(output_path, "w") if output_path else None

    try:
        for i, unit in enumerate(units):
            if (i + 1) % 50 == 0:
                log.info("Judge progress: %d/%d (%.0f%%)", i + 1, len(units), (i + 1) / len(units) * 100)

            result = _judge_unit(client, unit, model)

            if result is None:
                errors += 1
                continue

            score = result["score"]
            score_counts[score] = score_counts.get(score, 0) + 1

            if f_out:
                f_out.write(json.dumps({
                    "source": unit.source,
                    "domain": unit.domain,
                    "unit_type": unit.unit_type,
                    "regex_score": unit.quality_score,
                    "judge_score": score,
                    "reason": result.get("reason", ""),
                }) + "\n")

            if score >= min_score:
                passed.append(unit)
            else:
                rejected += 1
    finally:
        if f_out:
            f_out.close()

    log.info(
        "Judge results: %d passed, %d rejected, %d errors. Score distribution: %s",
        len(passed), rejected, errors,
        " ".join(f"{k}★={v}" for k, v in sorted(score_counts.items())),
    )
    return passed
