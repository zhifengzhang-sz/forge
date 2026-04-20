# Event-Sourcing Pattern Seeds

Fifteen distinct event-sourcing patterns used as the backbone for v1 Phase
C ES synthesis. Each entry lists: a one-line summary, the key API surface
the pattern exercises, and the must-haves a correct answer needs. Subagents
pick a pattern plus a phrasing template from `phrasings.md` and produce a
typed, compile-clean snippet.

House rules for every pattern:

- **Pure TypeScript.** No specific event-store client dependency required at
  compile time (the v1 verifier probe installs only `@types/node` and
  `typescript`). Type the event-store interface as a local `interface`
  instead of importing from `@eventstore/db-client`.
- **Typed discriminated unions for Command, Event, and State.** Every
  event and command carries a `type: "..."` literal; the state (if not a
  simple record) carries a `status: "..."` literal. Exhaustiveness in
  `evolve` / `decide` is enforced by a `never` default.
- **Pure `evolve(state, event): State`.** No I/O, no `Date.now()`, no
  random IDs. Event carries any clock/identity it needs.
- **`decide(command, state): Event[] | Event`** — returns the event(s) the
  command produces, or throws a typed domain error. No I/O.
- **Initial state is a literal**, not a constructor with side effects.
- **NEVER import from `"xstate"`** in an ES answer (shape guard — the
  verifier will reject).
- **NEVER use `setup(...)`** — that's XState. ES answers compose pure
  functions, not state-machine factories.
- Effect-TS is allowed when wrapping an event-store IO boundary
  (`Effect.gen` + event store calls is a valid idiom). It is NOT allowed
  in the pure `evolve` / `decide` core.
- Avoid the canonical aggregate names from the eval set: **ShoppingCart,
  BankAccount, CartSummary, CartCreated, ItemAdded, ItemRemoved, Deposit,
  Withdraw**. Use substitutes (Invoice, Reservation, Order, GroupCheckout,
  Shipment, CashRegister, GuestStay, Subscription, LoyaltyAccount, etc.).

---

1. **Pure Decider** — encapsulates `decide`, `evolve`, `initialState` for
   one aggregate. No I/O. Deterministic given the same event stream.
   - API: `type Decider<Command, Event, State>`, `initialState: State`,
     `decide(command, state): Event[]`, `evolve(state, event): State`.
   - Must include: exported `interface Decider<C, E, S>` (or a record
     matching that shape), one worked aggregate (Reservation / Invoice /
     Shipment), a `decide` that rejects at least one invalid command with
     a typed error, an exhaustive `evolve` with a `never` default.

2. **evolve as a pure reducer** — fold over an event stream to reconstruct
   state.
   - API: `evolve(state: State, event: Event): State`, discriminated
     `Event` union, `events.reduce(evolve, initialState)`.
   - Must include: typed `Event` DU, exhaustive switch on `event.type`,
     default branch with `const _: never = event`, a helper like
     `aggregate(events: Event[]): State` that does the reduce.

3. **decide with domain guard** — a command-handler function that throws
   typed errors on invariant violation.
   - API: `decide(command: Command, state: State): Event[]`, typed
     `class InvariantError extends Error` (or a discriminated union of
     errors), exhaustive switch on `command.type`.
   - Must include: at least two commands, at least one invariant check
     that throws a typed error (e.g., cannot cancel a completed order),
     exhaustive switch with `never` default, return type `Event[]` or a
     single `Event`.

4. **Projection / Read Model** — fold a specific event subset into a
   flat, query-friendly view. Distinct from `evolve` which reconstructs
   aggregate state.
   - API: `project(view: View, event: Event): View`, a view type with
     denormalised fields, an initial `emptyView`.
   - Must include: `View` type, `emptyView` literal, a `project` that
     switches on `event.type` and returns an updated `View`, ignoring
     events outside the view's interest (return the view unchanged).
     Exhaustive matching on the events it DOES handle.

5. **Snapshot + versioned upgrade** — serialize state with a version tag;
   on load, run upgrade functions if the stored version is older than
   the current schema.
   - API: `type Snapshot<S> = { version: number; state: S }`,
     `upgradeSnapshot(snap: Snapshot<unknown>): Snapshot<CurrentState>`.
   - Must include: a `CURRENT_VERSION` constant, at least two upgrader
     functions keyed by source version (e.g., `v1Tov2`, `v2Tov3`),
     a loader that runs upgraders in order, and a type assertion that
     the post-upgrade value matches `CurrentState`.

6. **Optimistic concurrency with expectedRevision** — appendToStream
   guarded by the last-known stream revision; surface a typed
   `ConcurrencyConflict` when the server reports a mismatch.
   - API: `appendToStream(stream, events, { expectedRevision })`,
     local `interface EventStore { appendToStream; readStream }`,
     `class ConcurrencyConflict extends Error`.
   - Must include: the local `EventStore` interface typed with
     `expectedRevision: bigint | "no_stream" | "any"`, a write path
     that reads current revision, appends with that revision, and
     converts the store's conflict signal into `ConcurrencyConflict`.

7. **Subscription from checkpoint** — a projection runner that resumes
   from the last persisted checkpoint after a restart.
   - API: `interface CheckpointStore { read(name): Promise<bigint | null>; write(name, pos): Promise<void> }`,
     an async iterator over events, a loop that applies each event then
     writes the checkpoint.
   - Must include: the checkpoint-store interface, a `runSubscription`
     function that reads the last checkpoint, consumes events, applies
     them to a read model, and writes the new checkpoint. Crash-safe:
     write must happen AFTER the projection has been applied.

8. **CommandHandler over an EventStore** — load current state, `decide`
   on the command, append resulting events with expected revision.
   - API: `handle(command, { streamName, eventStore }): Promise<void>`,
     a local `interface EventStore`, the Decider trio.
   - Must include: load path (`readStream` + `reduce(evolve, initial)`),
     `decide(command, state)` call, `appendToStream(streamName, events,
     { expectedRevision })` call. No business logic in the handler — it
     orchestrates, it does not decide.

9. **Read stream with aggregateStream helper** — generic folder that
   replays a stream into an aggregate state.
   - API: `async function aggregateStream<S, E>(stream: AsyncIterable<E>,
     evolve: (s: S, e: E) => S, initial: S): Promise<S>`.
   - Must include: a generic async function with two type parameters, a
     `for await` loop, typed `initial` state, return of the folded
     state. Must not mutate the input; accumulate with let + reassign
     or a local reduce helper.

10. **Integration event translation** — translate a domain event into an
    integration event posted to an outbox. Distinct shape; domain
    internals stay internal.
    - API: a `translate(domainEvent): IntegrationEvent | null` pure
      function, `type IntegrationEvent = { schema: string; payload: ... }`.
    - Must include: a `translate` that is a pure function of the domain
      event, a discriminated `IntegrationEvent` with a versioned
      `schema` field, at least one domain event that maps to `null`
      (not every domain event is an integration event).

11. **Outbox pattern** — atomically persist domain events plus the
    integration payload in one transaction; a separate worker publishes
    the outbox row.
    - API: `async function appendWithOutbox(stream, events, outboxRows,
      conn)`, a local `interface OutboxRow { id; payload; publishedAt }`.
    - Must include: a transactional function that inserts outbox rows
      and appends to the stream in one unit of work (use a local
      transactional interface; do not pull in `pg` / `mysql2`), a
      commentary comment describing the publisher's role (poll + mark
      published) without actually implementing it.

12. **Process Manager (saga) skeleton** — reacts to one event, emits
    the next command. Stateless variant — state lives in the stream it
    subscribes to.
    - API: `react(event: Event): Command | null`, or a state machine
      encoded as a pure function `(state, event) => { nextState,
      command? }`.
    - Must include: pure `react` function, a `null` return for events
      the process manager doesn't care about, a worked 2–3 step saga
      (e.g., OrderConfirmed → ReserveStock → StockReserved →
      ChargePayment).

13. **Upcasting** — migrate an old event shape to the current one at
    read time, without rewriting the stream.
    - API: `upcast(raw: unknown): Event`, per-type handlers that pattern
      match on `raw.type` and a `version` field.
    - Must include: a raw/stored event union with an older shape, a
      current event union, an `upcast` function that switches on
      `{ type, version }` and returns the current shape, a
      default that throws `UnknownEventError`.

14. **Read model rebuild** — wipe and re-project from revision 0, with
    progress reporting and a resumable chunked loop.
    - API: `rebuild({ from: bigint; batchSize: number; project; persist
      }): Promise<void>`, a local event-store iterator with revision
      bookkeeping.
    - Must include: a loop that reads events in batches, calls
      `project(view, event)` per event, persists the batch to the read
      store, and advances a revision cursor. A `from` option lets
      rebuilds resume after a crash.

15. **CQRS boundary** — a minimal command/query split. Writes go through
    `decide` + `appendToStream`. Reads go through a read model populated
    by a projection. The two sides share the event DU but NOT the state
    type.
    - API: `handleCommand`, `queryView`, two independent functions.
    - Must include: a `handleCommand(cmd): Promise<void>` that uses
      Decider + EventStore, a `queryView(id): Promise<View | null>`
      that reads from a read store, and a comment explaining that the
      read store is eventually consistent with the event store. No
      shared state type across the boundary.
