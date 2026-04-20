# fp-ts / Effect-TS Pattern Seeds

Fifteen distinct fp-ts (2.x) / Effect-TS (3.x) patterns used as the backbone
for v0.7 Phase B/C FP synthesis. Each entry lists: a one-line summary, the
key API surface the pattern exercises, and the must-haves a correct answer
needs. Subagents pick a pattern plus a phrasing template from `phrasings.md`
and produce a typed, compile-clean snippet.

House rules for every pattern:

- **Effect imports come from `"effect"` (v3), not `"@effect/io"` or `"@effect/data"`** — those were v2 paths. Example: `import { Effect, Layer, Context, Schedule } from "effect"`.
- **fp-ts imports stay on v2**: `import { pipe } from "fp-ts/function"`, `import * as O from "fp-ts/Option"`, `import * as E from "fp-ts/Either"`, `import * as TE from "fp-ts/TaskEither"`.
- **NEVER import from `"xstate"`** in an FP answer (XState-leakage guard; will be rejected by the FP verifier).
- **NEVER use `setup(...)`** — that is an XState shape. FP answers use `Effect.gen`, `pipe`, `Layer`, `Context.GenericTag`, etc.
- Prefer `Effect.gen(function* () { ... })` with `yield*` (v3) over `Effect.flatMap` chains when there are ≥ 2 sequential effects. Single effects can stay as `Effect.map` / `Effect.tap`.
- Error channels are typed — errors are tagged classes or `Data.TaggedError` subclasses, not raw `Error`. Use `Effect.catchTag(tag, handler)` to discriminate.
- Prefer typed services via `Context.GenericTag<Service>("scope/Service")` (v3). The old string-tag form and `Context.Tag<Service>()` without identifier are discouraged.

---

1. **Effect.gen with `yield*`** — sequential effect composition via generator.
   - API: `Effect.gen`, `yield*` on `Effect<A, E, R>`.
   - Must include: `Effect.gen(function* () { const a = yield* eff1; const b = yield* eff2(a); return b })`. No `.pipe(Effect.flatMap(...))` chain when `gen` is cleaner. Typed error channel preserved across yields.

2. **Layer composition** — build a service graph from `Layer.succeed` / `Layer.effect` and wire it with `Layer.merge` / `Layer.provide`.
   - API: `Layer.succeed(Tag, impl)`, `Layer.effect(Tag, Effect)`, `Layer.merge(a, b)`, `Layer.provide(innerLayer)`, `Effect.provide(program, MainLayer)`.
   - Must include: at least one `Layer.succeed` and one `Layer.effect`, one `Layer.merge` (or `Layer.mergeAll`), the final `Effect.provide(program, LiveLayer)` call, and a comment showing the dependency direction.

3. **Context.GenericTag service** — declare a service and resolve it inside `Effect.gen`.
   - API: `Context.GenericTag<Service>("scope/Service")`, `yield* ServiceTag` to access it.
   - Must include: an `interface Service { ... }`, `const Service = Context.GenericTag<Service>("app/Service")`, a program that does `const s = yield* Service` and calls a method on it, and a `Layer.succeed(Service, { ... })` wiring.

4. **Tagged errors + Effect.catchTag** — typed error channel with discriminated union handling.
   - API: Either `class FooError { readonly _tag = "FooError" as const; constructor(readonly msg: string) {} }` or `class FooError extends Data.TaggedError("FooError")<{ msg: string }> {}`. `Effect.catchTag("FooError", handler)`.
   - Must include: at least two distinct `_tag` error classes, a program that can fail with either, `Effect.catchTag` resolving one tag, and the remaining error type still showing in the `E` channel of the resulting `Effect<A, E, R>`.

5. **Schedule retry** — retry a failing effect with exponential backoff + jitter + cap.
   - API: `Schedule.exponential(Duration)`, `Schedule.jittered`, `Schedule.compose`, `Schedule.intersect` (or `Schedule.upTo` / `Schedule.recurs`), `Effect.retry(effect, schedule)`.
   - Must include: a composed schedule (exponential base + jitter + bounded recurs/duration), application via `Effect.retry(eff, schedule)`, and a comment on what makes the schedule bounded.

6. **Effect.all — sequential vs parallel** — combine N effects with explicit concurrency mode.
   - API: `Effect.all(effects, { concurrency: "unbounded" | number | "inherit" })`, default is sequential.
   - Must include: one `Effect.all(effects)` example (sequential, default) and one `Effect.all(effects, { concurrency: "unbounded" })` (or a numeric concurrency limit). Typed tuple return preserved.

7. **Option chain** — nullable-safe lookup chain with `fp-ts/Option`.
   - API: `O.fromNullable`, `O.chain`, `O.map`, `O.getOrElse`, `pipe`.
   - Must include: `pipe(input, O.fromNullable, O.chain(step), O.map(transform), O.getOrElse(() => fallback))` with types flowing through. No `.pipe()` chained on Option instances (fp-ts Option doesn't carry `.pipe` — use the standalone `pipe` function).

8. **Either chain** — synchronous error-or-value chain with `fp-ts/Either`.
   - API: `E.right`, `E.left`, `E.map`, `E.chainW` (widening chain), `E.mapLeft`, `E.fold` (or `E.match`), `pipe`.
   - Must include: a pipeline with at least one `E.chainW` that widens the left type, a `E.mapLeft` that transforms the error, and a final `E.fold` / `E.match` collapsing to a single return type.

9. **TaskEither.tryCatch with error narrowing** — lift a throwing async function into `TaskEither<E, A>`.
   - API: `TE.tryCatch(() => Promise<A>, (u) => E)`, typed error narrower for `unknown`.
   - Must include: `TE.tryCatch(async () => ..., (u) => u instanceof Error ? u : new Error(String(u)))` (or a tagged error class) — the onError callback must narrow `unknown` to the declared `E`, never just cast.

10. **pipe composition** — the `pipe(value, ...fns)` pattern as the canonical data-first application form.
    - API: `import { pipe } from "fp-ts/function"` or `import { pipe } from "effect"`.
    - Must include: at least one pipeline with 3+ steps, demonstrating that `pipe(a, f, g, h)` reads top-to-bottom the same as data flows. Show one example where `pipe` is clearly superior to nested calls (`h(g(f(a)))`).

11. **Branded types** — nominal typing via a brand tag to prevent primitive confusion.
    - API (Effect v3): `import { Brand } from "effect"`. `type Email = string & Brand.Brand<"Email">`. Constructor via `Brand.nominal<Email>()` or `Brand.refined<Email>(pred, onInvalid)`.
    - Must include: declaration of the branded type, a constructor (`nominal` for type-only, `refined` for runtime-validated), and a function signature that takes `Email` (not `string`) to show the compile-time safety.

12. **Validation accumulation** — collect ALL validation errors, not just the first.
    - API: fp-ts: `ReadonlyNonEmptyArray<E>` + `E.getApplicativeValidation(semigroup)` or `Apply.sequenceT`. Effect v3: `Effect.validateAll` / `Effect.partition` / `Effect.forEach(..., { discard: false })` combined with error accumulation.
    - Must include: at least three field-level validators that each return `Either<E, A>` (or `Effect<A, E>`), combined so failures accumulate into a `ReadonlyNonEmptyArray<E>` (fp-ts) or reported via `Effect.validateAll` (Effect v3). Do NOT short-circuit on the first error.

13. **Effect.scoped for resource acquire/release** — safe resource lifecycle via `Scope`.
    - API: `Effect.acquireRelease(acquire, release)`, `Effect.scoped(program)`, `Scope.Scope` (implicit in the R channel until `scoped` closes it).
    - Must include: `Effect.acquireRelease(acquire, (r) => release(r))` returning an `Effect<Resource, E, Scope>`, use of the resource inside `Effect.gen`, and a wrapping `Effect.scoped(...)` call that guarantees the release runs on both success and failure.

14. **Effect.fork + Effect.race** — concurrency primitives for fibres and racing.
    - API: `Effect.fork` (returns `Fiber<A, E>`), `Fiber.join`, `Fiber.interrupt`, `Effect.race(a, b)`, `Effect.raceAll`.
    - Must include: one example of `Effect.fork` producing a fiber that is later joined or interrupted, and one example of `Effect.race(primary, timeout)` where `timeout` uses `Effect.sleep(Duration.seconds(n))` + failure. Show that the loser is interrupted automatically.

15. **Effect.tap for side effects** — run an observable side effect without changing the pipeline value.
    - API: `Effect.tap(effect, (a) => sideEffect(a))`, `Effect.tapError`, `Effect.tapBoth`.
    - Must include: `pipe(program, Effect.tap((a) => Effect.logInfo(...)))` (or `Console.log`) where the returned effect is still `Effect<A, E, R>` — the tap's return is discarded. Contrast with `Effect.flatMap` in a one-line comment so the student sees why `tap` is the right tool for logging.
