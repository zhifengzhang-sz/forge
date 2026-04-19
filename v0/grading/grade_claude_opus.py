"""Grade Claude Opus 4.7 baseline (fresh subagent, no v0 context)."""
import json
from pathlib import Path

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")

GRADES = {
    "xstate-01": (5, "Full v5 idiom: setup({types, actors, guards}).createMachine. Discriminated union events. fromPromise<Output, Input> with proper input mapper. Hierarchical email>editing/validating states. guard: keyword. event.output. Cross-state target via #signup.password absolute ID."),
    "xstate-02": (5, "setup() with types/actions/guards. Discriminated event union. Guard isolates PAUSE structurally + via canPause guard. Actions named in setup, referenced by string."),
    "xstate-03": (5, "setup() with actors. fromPromise<User, {userId}>. invoke at state level (correct). onDone/onError with assign destructured. event.output (v5)."),
    "xstate-04": (5, "fromObservable<Output, Input> properly typed. Returns rxjs Observable<WsEvent>. Auto-forwards events. setup() with actors. assign destructured."),
    "xstate-05": (5, "Perfect v4→v5 conversion. setup({types}).createMachine(). Destructured ({context}) => context.count + 1. Notes the three key v5 changes accurately."),
    "fp-01": (5, "Clean fp-ts: pipe + E.chain composition. Branded Email type. Correct namespace import. Note about Validation for accumulating errors."),
    "fp-02": (5, "Idiomatic Effect-TS: Effect.gen with current API (no $ adapter). Context.GenericTag. Effect.catchTag('UserNotFound', ...). Layer.succeed. Effect.provide (not deprecated provideLayer). class with _tag for error type."),
    "fp-03": (5, "pipe + O.chain + O.map composition. O.fromNullable for optional field. Correct namespace usage. Concise rule-of-thumb explanation."),
    "fp-04": (4, "Correct TE.chain composition. Minimal — just shows the pipe; could include error handling example."),
    "fp-05": (5, "Idiomatic Effect-TS Layer composition. Context.GenericTag for both. Layer.succeed for Logger. Layer.effect for DB-depending-on-Logger. Layer.provideMerge for wiring. Effect.gen with current API throughout."),
    "rx-01": (5, "debounceTime + distinctUntilChanged + filter + switchMap. Properly typed. Note about AbortController for true cancellation."),
    "rx-02": (5, "combineLatest + map. Strongly typed ViewModel + interfaces. Note about combineLatest waiting semantics."),
    "rx-03": (5, "mergeMap with concurrency 3. Concise. Notes the queue behavior."),
    "rx-04": (5, "Clear distinction. Side-by-side wrong/right code. Bonus: ReplaySubject(n), getValue() synchronous read."),
    "rx-05": (5, "tap() with object form (next/error/complete). OperatorFunction<T, T> properly typed. Note about type inference in pipe chains."),
    "es-01": (5, "Discriminated union State (status: 'empty'|'open'). Discriminated union events. Pure evolve with switch. Type-narrows on state.status."),
    "es-02": (5, "Generic Decider<S,C,E> interface — clean abstraction. Discriminated commands and events. Correct rejection logic. Pure functions."),
    "es-03": (5, "Generic makeCommandHandler with Decider + EventStore injection. Load → Rebuild (reduce) → Decide → Append. Optimistic concurrency via expectedVersion. Custom ConcurrencyError. Note about retry loops."),
    "es-04": (5, "project(state, event) handles undefined initial state. Math.max(0, ...) to floor. Note about denormalizing qty+price into ItemRemoved (real ES design tradeoff)."),
    "es-05": (5, "Excellent comparison table. TypeScript signature with EventStore + SnapshotStore + schemaVersion. Working implementation. Rule of thumb for when to add snapshots."),
    "cap-01": (5, "Correct iterative fib with type hints, docstring, ValueError on negative. Returns `a` after n iterations (correct algorithm)."),
    "cap-02": (5, "ACTUALLY USED THE READ TOOL. Honestly reported file doesn't exist. Inferred project type from context. Offered alternative manifest files. Perfect tool-use behavior."),
    "cap-03": (5, "All 4 topics covered with specific examples (ETag, Cache-Control, @deprecated, GraphiQL, codegen). Concrete recommendation."),
    "cap-04": (5, "Test-pin first (excellent practice). Smallest-seam-first incremental approach. Optional follow-up. No code as instructed."),
}

raw = json.loads((V0 / "claude_opus.raw.json").read_text())
eval_data = json.loads(Path("/home/zzhang/dev/ai/models/forge/v0/eval/v0.json").read_text())
domain_lookup = {p["id"]: p["domain"] for p in eval_data["prompts"]}
graded = {**raw, "graded": []}
for r in raw["responses"]:
    score, rationale = GRADES.get(r["id"], (None, ""))
    graded["graded"].append({**r, "domain": domain_lookup.get(r["id"]), "score": score, "rationale": rationale})
(V0 / "claude_opus.json").write_text(json.dumps(graded, indent=2))

# Compare all 4 models
print(f"{'arm':<22}{'domain_avg':>12}{'cap_avg':>10}{'xstate_avg':>12}")
print("-" * 60)
for arm in ["base (qwen3:14b)", "curated", "extracted", "claude-opus-4-7"]:
    if arm == "base (qwen3:14b)":
        graded_ = json.loads((V0 / "base.json").read_text())["graded"]
    elif arm == "claude-opus-4-7":
        graded_ = json.loads((V0 / "claude_opus.json").read_text())["graded"]
    else:
        graded_ = json.loads((V0 / f"{arm}.json").read_text())["graded"]
    domain = [g["score"] for g in graded_ if g["domain"] != "capability" and g["score"]]
    cap = [g["score"] for g in graded_ if g["domain"] == "capability" and g["score"]]
    xstate = [g["score"] for g in graded_ if g["domain"] == "xstate" and g["score"]]
    print(f"{arm:<22}{sum(domain)/len(domain):>12.2f}{sum(cap)/len(cap):>10.2f}{sum(xstate)/len(xstate):>12.2f}")

print("\nPer-domain breakdown for Claude Opus 4.7:")
graded_ = json.loads((V0 / "claude_opus.json").read_text())["graded"]
for d in ["xstate", "fp", "reactive", "eventsourcing", "capability"]:
    s = [g["score"] for g in graded_ if g["domain"] == d and g["score"]]
    print(f"  {d}: avg={sum(s)/len(s):.2f}, scores={s}")
