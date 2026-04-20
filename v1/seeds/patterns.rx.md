# RxJS Pattern Seeds

Ten distinct RxJS (v7+) patterns used as the backbone for v0.7 Phase B/C RX
synthesis. Each entry lists: a one-line summary, the key API surface the
pattern exercises, and the must-haves a correct answer needs. Subagents
pick a pattern plus a phrasing template from `phrasings.md` and produce a
typed, compile-clean snippet.

House rules for every pattern:

- **All imports from `"rxjs"` or `"rxjs/operators"`** (v7 hoisted most operators into the root `"rxjs"` entry — either import path is fine, prefer `"rxjs"` for operators).
- **NEVER import from `"xstate"`** in an RX answer (XState-leakage guard; will be rejected by the RX verifier).
- **NEVER use `setup(...)`** — that is an XState shape. RX answers use `pipe`, operators, `Subject`-family, `Observable<T>`.
- **Do not use `.do(...)`** — that was a pre-v7 operator name. The current operator is `tap(...)`.
- **Do not use the deprecated `retryWhen`** unless the prompt explicitly asks for v6 compatibility. v7+ prefers `retry({ delay: (error, retryCount) => ... })`.
- Prefer the callable `pipe()` form on observables: `source$.pipe(opA(), opB(), opC())`. Keep type parameters explicit on subjects (`new BehaviorSubject<UserState>(initial)`).
- Observable stream variable names should end with `$` (`value$`, `click$`). Not required but improves training signal consistency.

---

1. **Typeahead: debounceTime + distinctUntilChanged + switchMap** — the canonical search-box pattern.
   - API: `fromEvent`, `map`, `debounceTime(ms)`, `distinctUntilChanged()`, `switchMap(q => httpSearch$(q))`, `filter`.
   - Must include: an input stream derived from user keystrokes (or `Observable<string>`), `debounceTime(300)` (or similar), `distinctUntilChanged()` to suppress duplicate emissions, and `switchMap` to cancel in-flight searches when a new query arrives. Short comment noting `mergeMap` here would be wrong (races).

2. **combineLatest with typed ViewModel** — derive a typed UI state from N source streams.
   - API: `combineLatest({ a: a$, b: b$ })` (object form, v7+) or `combineLatest([a$, b$]).pipe(map(([a, b]) => ...))`, `map` to a typed `ViewModel`.
   - Must include: at least two input streams with distinct types (e.g. `user$: Observable<User>`, `cart$: Observable<Cart>`), a `combineLatest` call (object form preferred), a `map` producing a typed `ViewModel`, and the `ViewModel` interface explicitly declared. First emission only fires once every source has emitted — note this in a comment.

3. **forkJoin — parallel completion** — run N independent observables in parallel and emit once all complete.
   - API: `forkJoin({ a: a$, b: b$, c: c$ })` (object form, v7+) or `forkJoin([a$, b$, c$])`.
   - Must include: at least three independent HTTP-like sources (each completing after one emission), `forkJoin` combining them, and a typed result destructured from the emission. Short comment: `forkJoin` requires every inner observable to complete — a `BehaviorSubject` or hot stream would hang it.

4. **mergeMap with concurrency limit** — bounded parallelism for outer events.
   - API: `mergeMap(project, concurrency)` where `concurrency: number` bounds the in-flight inner subscriptions.
   - Must include: an outer stream emitting N jobs, `mergeMap(job => processJob$(job), 4)` (or a similar numeric limit), and a comment noting the difference between `mergeMap` (interleaved results), `concatMap` (strict order, 1-at-a-time), and `switchMap` (cancels prior). Bounded concurrency is the specific reason to reach for `mergeMap` with a limit.

5. **BehaviorSubject — seeded + getValue** — stateful stream with an initial value and synchronous read.
   - API: `new BehaviorSubject<T>(initial)`, `.next(v)`, `.getValue()`, `.asObservable()`.
   - Must include: a typed `BehaviorSubject<State>(initialState)`, at least one `.next()` update, one `.getValue()` sync read (with a note that `getValue()` is a code-smell for reactive code — acceptable for one-shot reads like form-submit), and `.asObservable()` exposed to consumers so they can't call `.next` directly. Late subscribers receive the current value on subscription — state this explicitly.

6. **ReplaySubject — buffered replay** — emit the last N values to late subscribers.
   - API: `new ReplaySubject<T>(bufferSize, windowTime?)`.
   - Must include: a typed `ReplaySubject<T>(bufferSize)` (e.g. `new ReplaySubject<Event>(5)`), emissions before any subscriber exists, and a late subscriber that receives the buffered values on subscription. Contrast with `BehaviorSubject` (only the latest value) in a one-line comment.

7. **tap with object form `{ next, error, complete }`** — observer-shaped side effect.
   - API: `tap({ next: (v) => ..., error: (e) => ..., complete: () => ... })`.
   - Must include: at least one pipeline using `tap({ next, error, complete })` (all three handlers present), placed BEFORE the terminal subscription. Short comment: `tap` returns the same observable — it's for observation, not transformation. Do NOT use the legacy positional form `tap(next, error, complete)` (deprecated in v7).

8. **retry with backoff** — retry a failing inner observable with exponential delay.
   - API (v7+): `retry({ count, delay: (error, retryCount) => timer(ms) })`. Legacy `retryWhen` only if v6 compat is required.
   - Must include: `retry({ count: 3, delay: (err, retryCount) => timer(2 ** retryCount * 100) })` (or equivalent), surfacing the exhausted error to the consumer. One-line comment noting `retryWhen` is the v6 way and is superseded by `retry({ delay })`.

9. **Custom operator via pipe** — a reusable `OperatorFunction<T, R>` built from existing operators.
   - API: `import { OperatorFunction, pipe } from "rxjs"`. `export function logAndMap<T, R>(fn: (t: T) => R, tag: string): OperatorFunction<T, R> { return pipe(tap((t) => console.log(tag, t)), map(fn)); }`.
   - Must include: a function typed as `OperatorFunction<T, R>` (generic), composed via the standalone `pipe` from `"rxjs"` (NOT the method `.pipe` — that is on observables), returning `source$.pipe(...)` or a prebuilt pipe. Used downstream with `someSource$.pipe(myCustomOperator(...))`.

10. **shareReplay(n) for caching** — turn a cold observable into a replay-buffered hot one for multicast + cache.
    - API: `shareReplay({ bufferSize: 1, refCount: true })` (v7+, config form) or `shareReplay(1)` (shorthand).
    - Must include: a cold source (typically `ajax`, `fromFetch`, or a `defer`-wrapped promise), `shareReplay({ bufferSize: 1, refCount: true })` applied once, and two subscribers that demonstrate the source effect runs once. One-line comment on `refCount: true` releasing the upstream subscription when the last subscriber unsubscribes (avoids leaks).
