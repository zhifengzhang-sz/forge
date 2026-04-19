"""Grade v0.6 arm (fresh subagent, no v0 context)."""
import json
from pathlib import Path

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")

GRADES = {
    "xstate-01": (3, "setup() + types + actors + fromPromise typed correctly, but context.errors type mismatch (string[] vs object literals), NEXT guard logic is broken (fallthrough to password), uses legacy internal: false flag."),
    "xstate-02": (3, "setup() with types, assign for progress, but no guard keyword anywhere — PAUSE is gated only structurally by state placement; prompt explicitly asked for a guarded transition."),
    "xstate-03": (5, "Clean v5: setup({types, actors}), fromPromise<Profile,{id}> typed, invoke at state level with input mapper, onDone/onError destructured assign, event.output."),
    "xstate-04": (2, "fromObservable is imported but never called — wsInput is a bare Subscribable object. Root-level invoke mixed with states.open/closed that are unreachable. Broken machine shape."),
    "xstate-05": (5, "Perfect v4->v5: setup({types}).createMachine, destructured ({context}) => context.count+1, typed Ev union, uses 'TOGGLE': 'active' string shorthand."),
    "fp-01": (5, "Clean fp-ts: pipe + E.chain composition of three validators, tagged ValidationError union, branded Email wrapper, E.map to build final shape."),
    "fp-02": (1, "Fundamentally wrong domain: response is an XState machine, not Effect. No Effect.gen, no Layer, no yield*, no Context.Tag. Does not address the prompt at all."),
    "fp-03": (4, "Correct pipe + O.map + O.chain composition, but slightly over-engineered — O.chain with emptiness check is unnecessary since toUpperCase never fails; O.map would suffice."),
    "fp-04": (2, "Structure (pipe + TE.chain) is right but TE.fromPromise is not the fp-ts API (should be TE.tryCatch); inner lambdas use .then() which the prompt bans."),
    "fp-05": (1, "Fundamentally wrong domain: XState fromCallback/setup instead of Effect Layer. No Context.Tag, no Layer.merge, no Effect.gen. Off-topic."),
    "rx-01": (3, "Has debounceTime/distinctUntilChanged/switchMap and filter, but Observable type used without import, cast `as Observable<string>` and `map((results): { query; results }` produces a malformed TS type annotation."),
    "rx-02": (1, "Wrong library: imports {combine, assign, fromPromise} from xstate. No combineLatest, no RxJS map, no Observable<ViewModel>. Does not address the prompt."),
    "rx-03": (3, "Outer uploadStream uses mergeMap with config.concurrency — concurrency isn't hardcoded to 3, so the prompt's '3' isn't literal; inner uploadFile has broken AbortController-via-controller.signal that doesn't exist on the emitter."),
    "rx-04": (4, "Correct conceptual distinction with BehaviorSubject holding last value for late subscribers; clean code example, but Observable type referenced without import and no subscribe demonstration of late-subscriber semantics."),
    "rx-05": (3, "Uses tap correctly but generic type parameter T is lost — signature returns LogTap<unknown> instead of <T>(label) => OperatorFunction<T, T>; not genuinely type-safe across pipeline."),
    "es-01": (3, "Evolve-style reducer is named `shoppingCart` not `evolve`; ItemRemoved branch has a precedence bug: `state.totalCents - find(...)?.priceCents ?? 0` flattens to NaN. No exported initial state."),
    "es-02": (5, "Complete Decider: discriminated Command/Event/State, decide returns Event[], evolve is pure switch, initial state exported, rejects negative amount and overdraft with reason codes."),
    "es-03": (2, "Wrong approach — wraps the command handler in an XState machine instead of a plain function of (id, cmd, es) => events. Missing expectedRevision constant/type; events: any[] loses typing."),
    "es-04": (3, "Projector function is mostly correct with Math.max flooring, but `id: event.id` overwrites CartSummary.id on every event rather than preserving it; missing exported `ReadModel` alias."),
    "es-05": (2, "Prose tradeoff is OK but loadAggregate does not fold events on top of the snapshot (returns snapshot.state directly or full replay), doesn't take a store, and never invokes evolve."),
    "cap-01": (5, "Correct iterative Fibonacci with brief docstring, handles n<2 base case, loops n-1 times returning b — idiomatic and type-hinted."),
    "cap-02": (2, "Did not use the Read tool; emitted Deno TypeScript scripting code importing fs from deno.land, did not actually read package.json nor summarize deps."),
    "cap-03": (4, "Covers caching (HTTP headers vs single endpoint), over-fetching, schema evolution (deprecation directives, versioning), tooling (Swagger, GraphiQL); concrete but no explicit recommendation."),
    "cap-04": (4, "Three clear extraction steps (parse/validate/persist) emphasizing pure functions and adapter boundaries; mentions mockability but does not call out test-pin-first as an explicit guard."),
}

raw = json.loads((V0 / "v0.6.raw.json").read_text())
eval_data = json.loads(Path("/home/zzhang/dev/ai/models/forge/v0/eval/v0.json").read_text())
domain_lookup = {p["id"]: p["domain"] for p in eval_data["prompts"]}
graded = {**raw, "graded": []}
for r in raw["responses"]:
    score, rationale = GRADES.get(r["id"], (None, ""))
    graded["graded"].append({**r, "domain": domain_lookup.get(r["id"]), "score": score, "rationale": rationale})
(V0 / "v0.6.json").write_text(json.dumps(graded, indent=2))

# Compare all 5 models
print(f"{'arm':<22}{'domain_avg':>12}{'cap_avg':>10}{'xstate_avg':>12}")
print("-" * 60)
for arm in ["base (qwen3:14b)", "curated", "extracted", "claude-opus-4-7", "v0.6"]:
    if arm == "base (qwen3:14b)":
        graded_ = json.loads((V0 / "base.json").read_text())["graded"]
    elif arm == "claude-opus-4-7":
        graded_ = json.loads((V0 / "claude_opus.json").read_text())["graded"]
    elif arm == "v0.6":
        graded_ = json.loads((V0 / "v0.6.json").read_text())["graded"]
    else:
        graded_ = json.loads((V0 / f"{arm}.json").read_text())["graded"]
    domain = [g["score"] for g in graded_ if g["domain"] != "capability" and g["score"]]
    cap = [g["score"] for g in graded_ if g["domain"] == "capability" and g["score"]]
    xstate = [g["score"] for g in graded_ if g["domain"] == "xstate" and g["score"]]
    print(f"{arm:<22}{sum(domain)/len(domain):>12.2f}{sum(cap)/len(cap):>10.2f}{sum(xstate)/len(xstate):>12.2f}")

print("\nPer-domain breakdown for v0.6:")
graded_ = json.loads((V0 / "v0.6.json").read_text())["graded"]
for d in ["xstate", "fp", "reactive", "eventsourcing", "capability"]:
    s = [g["score"] for g in graded_ if g["domain"] == d and g["score"]]
    print(f"  {d}: avg={sum(s)/len(s):.2f}, scores={s}")
