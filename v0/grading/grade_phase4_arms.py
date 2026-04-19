"""Apply Phase 4 grades for both fine-tuned arms.

Grading is calibrated against the qwen3:14b base scores (same prompt set, same scale).
"""
import json
from pathlib import Path

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")

CURATED_GRADES = {
    "xstate-01": (1, "Final code uses INVENTED API: setup({...}).withConfig({}).states({...}). withConfig doesn't exist in v5; states() isn't a method. Plus context.validateEmailService.result.message references non-existent fields. Worse than base (was 2)."),
    "xstate-02": (2, "createMachine alone, no setup(). Has assign, but no real guard implementation. Same level as base."),
    "xstate-03": (1, "createMachine alone, entry: [invoke(...)] is invalid (invoke goes at state level, not in entry). (context, event) v4 signatures. interpret() v4. Same as base."),
    "xstate-04": (1, "createMachine, services: {websocketService: fromObservable(messages$)} — services is v4 keyword. (context, event) v4 signatures. Worse than base on key signals."),
    "xstate-05": (1, "STILL v4 conversion failure. Outputs `assign({ count: (ctx) => ctx.count + 1 })` — v4 callback signature. No setup(). Same as base."),
    "fp-01": (4, "Clean fp-ts: pipe + Either + E.chain. Imports `pipe from 'fp-ts/pipeable'` (deprecated path; should be 'fp-ts/function') — minor demerit but works."),
    "fp-02": (1, "Wrote SCALA / Cats Effect. `import cats.effect._`, IO, Layer.succeed. Same failure mode as base."),
    "fp-03": (3, "Clean Option.map composition. Imports Option from fp-ts/Option (works as namespace import). Code is correct."),
    "fp-04": (3, "Clean TE.chain composition with pipe. Correct types and shape."),
    "fp-05": (1, "Wrote SCALA / ZIO. `trait Logger`, `ZLayer.succeed`, `ZIOAppDefault`. Same as base."),
    "rx-01": (4, "debounceTime + distinctUntilChanged + switchMap. Includes `map` import (used in fromEvent) but doesn't import it. Minor flaw."),
    "rx-02": (4, "Clean combineLatest + map + ViewModel interface."),
    "rx-03": (4, "mergeMap with concurrency 3, includes from() for source."),
    "rx-04": (4, "BehaviorSubject vs Subject correct, with comparison code examples."),
    "rx-05": (4, "Type-safe OperatorFunction<T, T> using tap. Clean."),
    "es-01": (4, "TypeScript with discriminated union Event, type State, evolve switch with quantity-aware ItemAdded merge. Solid."),
    "es-02": (3, "Pattern correct (decide/evolve/initial). Has TypeScript interfaces declared but the code body labeled javascript. Logic correct."),
    "es-03": (2, "JavaScript implementation with correct optimistic-concurrency pattern (passes expectedVersion to appendToStream). But pure JS; still pseudo-codey."),
    "es-04": (2, "JavaScript, project(state, event) function correct logic with init handling. No TypeScript types."),
    "es-05": (4, "TypeScript. Tradeoffs covered. loadAggregate signature correct."),
    "cap-01": (5, "Correct iterative fib, type hints, full docstring."),
    "cap-02": (1, "HALLUCINATED file contents! Listed express, lodash, winston, eslint, jest with confidence. Same failure mode as base — fine-tuning did NOT fix this."),
    "cap-03": (3, "Covered all 4 topics. Adequate."),
    "cap-04": (4, "Clean 3-step plan with goals, actions, outcomes. No code. Slight bonus for richer structure."),
}

EXTRACTED_GRADES = {
    "xstate-01": (2, "Has the right outer structure: setup({types: {context, events}}).createMachine({...}). BUT internals broken: invoke src wraps fromPromise(validateEmailUniqueness(context.email)) — calls fromPromise eagerly with a Promise instead of providing a function. Uses cond: (v4) and target: (context) => ... (v5 doesn't allow function targets). Slight improvement over base structurally."),
    "xstate-02": (2, "createMachine alone, no setup(). Same level as base."),
    "xstate-03": (2, "createMachine, has fromPromise wrapped correctly inside src. Uses event.data (v4 — should be event.output). Mixed v4/v5."),
    "xstate-04": (2, "createMachine, fromObservable usage broken. (context, event) v4 signature."),
    "xstate-05": (1, "Outputs literal `Machine(...)` with `actions.assign({ count: (ctx) => ctx.count + 1 })` and claims 'assign is now part of actions module' — entirely fabricated. Worse than even just renaming."),
    "fp-01": (4, "Clean fp-ts: namespace * as O import, pipe + E.chain composition."),
    "fp-02": (1, "Wrote SCALA. `import cats.effect.kernel`, Effect.gen[Config]. Same as base."),
    "fp-03": (3, "Clean import * as O from 'fp-ts/Option'. Idiomatic Option.map composition. Correct."),
    "fp-04": (3, "Clean pipe + chain composition. Imports look right."),
    "fp-05": (2, "Wrote TypeScript! Uses interface Logger / interface Database with class-based DI. NOT Effect-TS Layer/Context.Tag — but at least it's the right language. Same as base."),
    "rx-01": (4, "Idiomatic typeahead with the three required operators."),
    "rx-02": (4, "Clean combineLatest + map + ViewModel interface."),
    "rx-03": (4, "mergeMap with concurrency 3, clean."),
    "rx-04": (4, "Correct conceptual difference, BehaviorSubject seeded with null."),
    "rx-05": (4, "Type-safe OperatorFunction<T, T> using tap."),
    "es-01": (4, "TypeScript with discriminated union, plus cartId validation invariant. Solid."),
    "es-02": (3, "Pattern correct, generates RejectWithdrawal event (extra) which evolve ignores. Pure JS but logic right."),
    "es-03": (2, "JavaScript with handleCommand that reads stream, rehydrates, applies command, appendToStream with currentVersion. Pseudo-code-ish but pattern correct."),
    "es-04": (2, "JavaScript, has cart-id mismatch validation. Logic correct."),
    "es-05": (4, "TypeScript. Tradeoffs covered. loadAggregate signature with EventStore/SnapshotStore."),
    "cap-01": (5, "Correct iterative fib with explicit `new_val = a + b` style. Full docstring."),
    "cap-02": (1, "HALLUCINATED! Listed express, react, lodash, webpack, eslint without any tool call. Same failure as base."),
    "cap-03": (3, "Covered all 4 topics adequately."),
    "cap-04": (4, "Clean 3-step plan with goals/actions/outcomes."),
}

for arm_name, GRADES in [("curated", CURATED_GRADES), ("extracted", EXTRACTED_GRADES)]:
    raw = json.loads((V0 / f"{arm_name}.raw.json").read_text())
    graded = {**raw, "graded": []}
    for r in raw["responses"]:
        score, rationale = GRADES.get(r["id"], (None, ""))
        graded["graded"].append({**r, "score": score, "rationale": rationale})
    (V0 / f"{arm_name}.json").write_text(json.dumps(graded, indent=2))

# Compute summary
print(f"\n{'arm':<12}{'domain_avg':>12}{'cap_avg':>10}{'xstate_avg':>12}")
print("-" * 50)
base = json.loads((V0 / "base.json").read_text())
for arm in ["base", "curated", "extracted"]:
    if arm == "base":
        graded = base["graded"]
    else:
        graded = json.loads((V0 / f"{arm}.json").read_text())["graded"]
    domain = [g["score"] for g in graded if g["domain"] != "capability" and g["score"]]
    cap = [g["score"] for g in graded if g["domain"] == "capability" and g["score"]]
    xstate = [g["score"] for g in graded if g["domain"] == "xstate" and g["score"]]
    print(f"{arm:<12}{sum(domain)/len(domain):>12.2f}{sum(cap)/len(cap):>10.2f}{sum(xstate)/len(xstate):>12.2f}")

print()
print("Per-domain breakdown:")
for arm in ["base", "curated", "extracted"]:
    if arm == "base":
        graded = base["graded"]
    else:
        graded = json.loads((V0 / f"{arm}.json").read_text())["graded"]
    print(f"\n  {arm}:")
    for d in ["xstate", "fp", "reactive", "eventsourcing", "capability"]:
        s = [g["score"] for g in graded if g["domain"] == d and g["score"]]
        print(f"    {d}: avg={sum(s)/len(s):.2f}, scores={s}")
