"""Template-based instruction generation from extracted code units.

Generates training instructions without calling an external API.
Each instruction is constructed from:
  - The domain (fp, reactive, xstate, eventsourcing)
  - The unit type (function, type)
  - Key identifiers extracted from the code (function names, type names)
  - Type references found in the code (Option, Either, Observable, etc.)

This is the free alternative to lib/common/instruct.py (which uses Claude API
for higher-quality, more varied instructions at ~$3-5 per run).

Usage:
    python3 -m lib.common.template_instruct
"""

import json
import re
import logging
from pathlib import Path

from lib.common.types import Unit

log = logging.getLogger(__name__)


def _extract_name(code: str) -> str:
    """Pull the primary exported/defined name from the code."""
    for m in re.finditer(r'export\s+(?:function|const|type|interface)\s+(\w+)', code):
        return m.group(1)
    for m in re.finditer(r'(?:const|let)\s+(\w+)\s*=\s*(?:createMachine|setup)', code):
        return m.group(1)
    return ""


def _extract_types_hint(code: str) -> str:
    """Extract domain type names used in the code for instruction context."""
    types = set()
    for m in re.finditer(
        r'\b(Option|Either|Task|Reader|Effect|Layer|Observable|Subject|'
        r'StateMachine|MachineContext|EventStore|Aggregate)\b', code
    ):
        types.add(m.group(1))
    return ", ".join(sorted(types)[:3]) if types else "generic types"


# Instruction templates per domain and unit type.
# {name} = primary identifier, {types} = domain types found in code.
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "typescript.fp": {
        "function": [
            "Implement the fp-ts function `{name}` that works with {types} using pipe/flow composition and proper type safety",
            "Write `{name}` as an fp-ts utility function with full generic type parameters and functional composition",
            "Create the fp-ts function `{name}` following fp-ts conventions with Either/Option error handling",
        ],
        "type": [
            "Define the fp-ts type `{name}` with full generic type parameters following fp-ts type-class conventions",
            "Write the TypeScript type alias `{name}` as used in the fp-ts ecosystem with proper variance annotations",
        ],
    },
    "typescript.reactive": {
        "function": [
            "Implement the RxJS operator `{name}` that transforms Observable streams with proper subscription lifecycle management",
            "Write the RxJS `{name}` operator function with correct MonoTypeOperatorFunction/OperatorFunction signature",
            "Create the RxJS operator `{name}` handling subscriber notification, error propagation, and completion",
        ],
        "type": [
            "Define the TypeScript type for the RxJS `{name}` with correct generic constraints for Observable transformations",
        ],
    },
    "typescript.xstate": {
        "function": [
            "Create an XState v5 state machine `{name}` using setup() with typed context, events, actors, and guards",
            "Implement `{name}` as an XState v5 machine with proper state transitions, actions, and typed event handling",
            "Write an XState v5 actor logic `{name}` using fromPromise or fromCallback with typed input and output",
        ],
        "type": [
            "Define the XState v5 types for `{name}` including MachineContext, EventObject, and ActorLogic generics",
        ],
    },
    "typescript.eventsourcing": {
        "function": [
            "Implement `{name}` for an event-sourced system with aggregate state evolution, command handling, and event stream operations",
            "Write the event sourcing function `{name}` following CQRS/ES patterns with typed events, commands, and projections",
            "Create `{name}` as an event sourcing utility handling EventStore read/append with proper stream versioning",
        ],
        "type": [
            "Define the event sourcing types for `{name}` including Aggregate, Command, and Event discriminated unions",
        ],
    },
}


def make_instruction(unit: dict) -> str:
    """Generate a training instruction for a code unit.

    Uses deterministic template selection based on the unit's fingerprint
    so the same unit always produces the same instruction.
    """
    domain = unit["domain"]
    unit_type = unit["unit_type"]
    code = unit["code"]

    name = _extract_name(code) or "the utility"
    types = _extract_types_hint(code)

    templates = TEMPLATES.get(domain, {}).get(unit_type, [])
    if not templates:
        domain_short = domain.split(".")[-1]
        templates = [f"Write a TypeScript {unit_type} `{{name}}` for the {domain_short} domain with proper type safety"]

    # Deterministic template selection from fingerprint
    idx = hash(unit.get("fingerprint", code)) % len(templates)
    return templates[idx].format(name=name, types=types)


def generate_training_data(
    units_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> int:
    """Generate training JSONL from extracted units.

    Args:
        units_path: Path to extracted_units.jsonl
        output_path: Path to write typescript_training.jsonl
        metadata_path: Path to write metadata.jsonl

    Returns:
        Number of training examples generated.
    """
    units = []
    with open(units_path) as f:
        for line in f:
            units.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w") as f_train, open(metadata_path, "w") as f_meta:
        for i, unit in enumerate(units):
            instruction = make_instruction(unit)

            completion = unit["code"]
            if unit["imports"]:
                completion = unit["imports"] + "\n\n" + completion

            example_id = f"{unit['fingerprint']}-{i:04d}"

            f_train.write(json.dumps({
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": completion},
                ],
                "id": example_id,
            }) + "\n")

            f_meta.write(json.dumps({
                "id": example_id,
                "domain": unit["domain"],
                "source": unit["source"],
                "unit_type": unit["unit_type"],
                "quality_score": unit["quality_score"],
            }) + "\n")

            count += 1

    log.info("Generated %d training examples -> %s", count, output_path)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(__file__).parent.parent.parent
    n = generate_training_data(
        units_path=root / "dataset" / "extracted_units.jsonl",
        output_path=root / "dataset" / "typescript_training.jsonl",
        metadata_path=root / "dataset" / "metadata.jsonl",
    )
    print(f"Done: {n} examples")
