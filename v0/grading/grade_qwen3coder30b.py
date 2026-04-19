"""Apply Claude-proposed grades to base.raw.json → base.json"""
import json
from pathlib import Path

GRADES = {
    "xstate-01": (3, "setup() block correct + typed events + fromPromise. BUT external actions attached via deprecated .withConfig({actions, guards}) — v4 pattern. send: any leaks. Hybrid v4/v5."),
    "xstate-02": (2, "Pure v4 code with v5 imports. cond: instead of guard:, assign((context, event) => ...) callback signature, guards at root level (should be in setup())."),
    "xstate-03": (3, "Destructured ({ event }) action signatures and fromPromise correct. But no setup(), uses interpret(...).start() (v4 — v5 is createActor)."),
    "xstate-04": (2, "Imports right but conceptually broken. fromObservable not used inside invoke. v4 (context, event) action signature. webSocketMachine.start() wrong. Architectural confusion."),
    "xstate-05": (1, "Total v5 conversion failure. Just renamed Machine→createMachine, kept v4 ctx => ctx.count + 1, no setup(), no destructured context. Then claims 'idiomatic v5'."),
    "fp-01": (4, "Clean idiomatic fp-ts. pipe + Either + E.chain composition. Types correct. Minor redundancy in alt version."),
    "fp-02": (2, "Recognizes Effect concepts but uses outdated API. Effect.gen(function*($) is old adapter. Effect.service / Effect.serviceOf / Effect.provideLayer all deprecated. TaggedError syntax wrong."),
    "fp-03": (3, "Idiomatic shape (pipe + Option.map). But imports `{ Option, some, none }` from 'fp-ts/Option' then calls Option.map — type-only import won't have .map method. Won't compile."),
    "fp-04": (4, "Clean TaskEither composition. TE.tryCatch + pipe + TE.chain. Types correct."),
    "fp-05": (1, "WROTE SCALA. import scala.util.{Try, Success, Failure}. trait/object/case class everywhere. Catastrophic language miss."),
    "rx-01": (4, "debounceTime + distinctUntilChanged + switchMap, EMPTY guard for empty query. Slightly weak on TS type annotations but solid solution."),
    "rx-02": (3, "Main combineLatest solution correct. But the merge → switchMap → combineLatest 'alternative' is bizarre; the third alternative is identical to the first. Padding."),
    "rx-03": (3, "Core mergeMap(fn, 3) correct. Four near-identical alternatives. Explanation incorrectly claims 'maintains order' (mergeMap doesn't). catchError used without import."),
    "rx-04": (4, "Correct conceptual difference, BehaviorSubject seeded with default, clean UserService example. Solid."),
    "rx-05": (4, "Type signature correct. Two redundant manual implementations + one clean tap-based version. The tap version is the right answer."),
    "es-01": (2, "Wrote JavaScript despite TS request. Class-based events instead of discriminated union. JSON.parse(JSON.stringify()) for deep copy. No TS types as asked."),
    "es-02": (3, "Pattern correct (decide/evolve/initial), logic correct (rejects over-balance withdrawal). But pure JS — no TypeScript types as requested."),
    "es-03": (1, "WROTE JAVA. import com.eventstore.dbclient.*; CompletableFuture<WriteResult>. Catastrophic language miss."),
    "es-04": (2, "Logic mostly right but pure JavaScript — no TypeScript types. project signature has no types. Some bonus event handling not asked for."),
    "es-05": (4, "Actually TypeScript. Tradeoffs covered well. loadAggregate signature handles snapshot + event replay. Solid."),
    "cap-01": (5, "Correct iterative fib with proper Python type hints + thorough docstring."),
    "cap-02": (3, "Honest refusal — model has no tools wired up via /api/generate, so cannot actually call Read. Correct behavior given the runtime, but doesn't truly test tool-call capability (would need /v1/chat/completions with tool defs)."),
    "cap-03": (4, "Covered all 4 requested topics (caching, over-fetching, schema evolution, tooling). Accurate. 4 sentences."),
    "cap-04": (4, "Clear 3-step plan, no code as instructed. Markdown headers (## Step 1) instead of numbered list — equally numbered. Auto-check false negative."),
}

V0 = Path("/home/zzhang/dev/ai/models/forge/v0/results")
raw = json.loads((V0 / "base.raw.json").read_text())

graded = {**raw, "graded": []}
for r in raw["responses"]:
    score, rationale = GRADES.get(r["id"], (None, ""))
    graded["graded"].append({**r, "score": score, "rationale": rationale})

(V0 / "base.json").write_text(json.dumps(graded, indent=2))

# Compute averages
domain_scores = [g["score"] for g in graded["graded"] if g["domain"] != "capability" and g["score"]]
cap_scores = [g["score"] for g in graded["graded"] if g["domain"] == "capability" and g["score"]]
print(f"\nDomain avg: {sum(domain_scores)/len(domain_scores):.2f} (n={len(domain_scores)})")
print(f"Capability avg: {sum(cap_scores)/len(cap_scores):.2f} (n={len(cap_scores)})")
print()
for domain in ["xstate", "fp", "reactive", "eventsourcing"]:
    s = [g["score"] for g in graded["graded"] if g["domain"] == domain and g["score"]]
    print(f"  {domain}: avg={sum(s)/len(s):.2f}, scores={s}")
