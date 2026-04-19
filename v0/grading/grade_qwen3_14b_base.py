"""Apply Claude-proposed grades to qwen3:14b base.raw.json"""
import json
from pathlib import Path

GRADES = {
    "xstate-01": (2, "setup() block exists but config is structurally wrong: initial/context/states placed INSIDE setup({}) instead of in .createMachine(). Then .createMachine() called with no args. Uses cond: (v4) and (context, event) signatures. fromPromise wrapped inside an arrow function — broken."),
    "xstate-02": (2, "No setup(). Has guard: keyword (good) but guard body references non-existent context.state.matches(). assign uses (context, event) v4 signature."),
    "xstate-03": (1, "Structurally broken: invoke is placed inside entry: [{...}] which is invalid. Uses event.data (v4 — v5 is event.output). interpret(...).onTransition (v4)."),
    "xstate-04": (2, "No setup(). services: instead of actors:. actions: at root instead of in setup(). v4 signatures throughout. interpret() not createActor."),
    "xstate-05": (1, "v4→v5 conversion failure even worse than qwen3-coder. KEEPS literal `Machine({...})` in output and claims 'v5 enforces array syntax for actions' (false). Just wraps assign in []."),
    "fp-01": (4, "Clean fp-ts: pipe + Either + E.chain composition with separate validators. Idiomatic."),
    "fp-02": (1, "WROTE SCALA / ZIO with `import zio._`, `ZLayer.succeed`, `case class`. Catastrophic language miss."),
    "fp-03": (3, "Imports `{ pipe, Option }` from 'fp-ts' (no subpath, type-only). Won't compile because Option here is a type, not the namespace with .map. Shape correct."),
    "fp-04": (3, "Uses TE.chain correctly but writes `TE.fetchUser(id)` — fetchUser isn't on the TE namespace. Confused namespace access. Concept right."),
    "fp-05": (2, "Wrote TypeScript (improvement) but uses class-based DI, NOT Effect-TS Layer/Context.Tag. Misses the framework entirely; gives generic dependency injection instead."),
    "rx-01": (4, "debounceTime + distinctUntilChanged + switchMap, correct types and imports."),
    "rx-02": (4, "combineLatest with ViewModel interface, properly typed."),
    "rx-03": (4, "mergeMap with concurrency 3, clean."),
    "rx-04": (4, "Correct conceptual difference, BehaviorSubject seeded with null, summary table."),
    "rx-05": (4, "Type-safe OperatorFunction<T, T>. Correct manual Observable subscription pattern with proper next/error/complete forwarding."),
    "es-01": (4, "TypeScript! Discriminated union for Event, type for State, evolve switch handling all 3 events with quantity-aware logic. Solid."),
    "es-02": (3, "Pattern correct (decide/evolve/initial), logic correct. Pure JS — no TypeScript types as requested."),
    "es-03": (2, "Wrote JavaScript (no Java this time, improvement). Class-based with expected-version logic present, but mostly pseudo-code with placeholder comments and no actual EventStore client integration."),
    "es-04": (2, "Logic correct (handles ItemAdded/Removed/QuantityUpdated/CartCleared) but pure JS — no TypeScript types."),
    "es-05": (4, "TypeScript. Tradeoffs covered. loadAggregate signature with EventStore + SnapshotStore + Aggregate interfaces. Implementation marked 'details would go here' but signature is correct."),
    "cap-01": (5, "Correct iterative fib with proper Python type hints and full docstring."),
    "cap-02": (1, "HALLUCINATED file contents! Claimed '12 production dependencies and 7 development tools' with React/Express/Axios/Jest/ESLint without any tool call. Worse than honest refusal — actively misleading."),
    "cap-03": (3, "Covered all 4 topics adequately. Slightly less clear than qwen3-coder but accurate."),
    "cap-04": (4, "Clear 3-step plan with bold headers, helper-function naming, no actual code as instructed."),
}

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")
raw = json.loads((V0 / "base.raw.json").read_text())

graded = {**raw, "graded": []}
for r in raw["responses"]:
    score, rationale = GRADES.get(r["id"], (None, ""))
    graded["graded"].append({**r, "score": score, "rationale": rationale})

(V0 / "base.json").write_text(json.dumps(graded, indent=2))

domain_scores = [g["score"] for g in graded["graded"] if g["domain"] != "capability" and g["score"]]
cap_scores = [g["score"] for g in graded["graded"] if g["domain"] == "capability" and g["score"]]
print(f"\nDomain avg: {sum(domain_scores)/len(domain_scores):.2f} (n={len(domain_scores)})")
print(f"Capability avg: {sum(cap_scores)/len(cap_scores):.2f} (n={len(cap_scores)})")
print()
for domain in ["xstate", "fp", "reactive", "eventsourcing"]:
    s = [g["score"] for g in graded["graded"] if g["domain"] == domain and g["score"]]
    print(f"  {domain}: avg={sum(s)/len(s):.2f}, scores={s}")
