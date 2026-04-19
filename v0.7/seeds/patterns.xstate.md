# XState v5 Pattern Seeds

Thirty distinct XState v5 patterns used as the backbone for Phase A synthesis.
Each entry lists: a one-line summary, the key v5 API surface the pattern
exercises, and the must-haves a correct v5 answer needs. Subagents should pick
a pattern plus a phrasing template from `phrasings.md` and produce a typed,
compile-clean v5 machine.

House rules for every pattern:

- Use `setup({ types: { context, events } }).createMachine({...})` — never raw
  `createMachine` without `setup` when types are involved, and never the v4
  `Machine(...)` factory.
- Use the object-destructured action/guard signature (`({ context, event })`),
  never positional `(ctx, event)`.
- Prefer referenced `actions` / `guards` / `actors` declared in `setup` when
  the pattern reuses logic.

---

1. **Counter** — Increment/decrement/reset over a numeric context.
   - API: `setup()`, `assign`, typed events.
   - Must include: typed `{ count: number }` context, `assign({ count: ({ context }) => ... })`, at least one reset path.

2. **Toggle** — Two-state on/off flip.
   - API: `setup()`, typed events.
   - Must include: `initial`, two states, single `TOGGLE` event switching targets. No context needed unless the prompt adds one.

3. **Fetch (promise actor)** — idle → loading → success/failure.
   - API: `fromPromise`, `invoke`, `onDone`, `onError`, typed `input`/`output`.
   - Must include: `fromPromise<Output, Input>` actor, `invoke.src` referenced from `setup.actors`, `input: ({ context }) => ...`, assign on both `onDone.event.output` and `onError.event.error`.

4. **Multi-step form** — Sequential steps with per-step validation and back navigation.
   - API: Hierarchical states, `assign` per field, guards in setup, `final` state.
   - Must include: compound states per step, typed events per field, BACK transitions, a final `done` state.

5. **Parallel states** — Two regions running simultaneously.
   - API: `type: 'parallel'`, nested `states`.
   - Must include: one root state with `type: 'parallel'` and two sibling regions, each with its own `initial` and transitions.

6. **Hierarchical (compound) states** — Nested states with internal transitions.
   - API: Nested `states`, `initial`, target paths like `'#machine.parent.child'` or relative.
   - Must include: a parent state with `initial` and at least two nested child states with transitions between them.

7. **fromCallback listener** — Wrap an event-emitter / DOM listener as an actor.
   - API: `fromCallback`, `sendBack`, `receive`, cleanup return.
   - Must include: `fromCallback(({ sendBack, receive }) => { ...; return () => cleanup(); })`, events sent back typed so the parent machine can handle them.

8. **fromObservable actor** — Stream values from an Observable into the machine.
   - API: `fromObservable`, typed `input`.
   - Must include: `fromObservable<Event, Input>(({ input }) => observable)`, parent handles emitted event types in `on:`.

9. **Spawn child actor** — Dynamically spawn a child via `assign`.
   - API: `spawnChild` action creator or `assign(({ spawn }) => ...)`, `ActorRefFrom`.
   - Must include: a context slot holding the spawned ref (typed `ActorRefFrom<typeof child>`), creation via `spawn` inside `assign`, optional `stop` on teardown.

10. **sendTo parent/child** — Address an event at a specific actor by id.
    - API: `sendTo`, `ActorRef`, `systemId` or spawned id.
    - Must include: `sendTo((_, params) => ref, { type: 'EVENT' })` or id-based form; show a child raising an event back to the parent via `sendParent` / `sendTo(({ self }) => self._parent, ...)` where relevant.

11. **emit events** — Machine emits typed side-channel events subscribers can react to.
    - API: `emit` action, `types.emitted`, `actor.on('EVENT', ...)`.
    - Must include: `types: { emitted: {} as {...} }` in setup, `emit({ type: '...' })` inside an action, and a short note about subscribing via `actor.on`.

12. **enqueueActions** — Conditionally enqueue actions in one place.
    - API: `enqueueActions`, `enqueue.assign`, `enqueue.raise`, `enqueue.sendTo`, `check`.
    - Must include: `enqueueActions(({ enqueue, check }) => { if (check('guard')) enqueue.assign({...}); enqueue.raise({...}); })`.

13. **after delays** — Timed transition after N ms.
    - API: `after: { 5000: 'target' }`, `delays` in setup for named delays.
    - Must include: `after` block on a state, optional named delay referenced from `setup.delays`, and a way to reset the timer (re-enter or internal transition — see pattern 19).

14. **always transitions** — Eventless transitions evaluated on entry / context change.
    - API: `always: [{ guard, target }, ...]`.
    - Must include: at least one `always` array with guarded targets, typically used for auto-advance when a computed condition is met.

15. **guards in setup** — Named, reusable guards declared centrally.
    - API: `setup({ guards: { ... } })`, referenced as strings in transitions.
    - Must include: at least two guards in `setup.guards`, each using `({ context, event }) => boolean`, referenced by string from transitions.

16. **output from final state** — Return a typed value when the machine completes.
    - API: `type: 'final'`, `output: ({ context }) => ...`, `types.output`.
    - Must include: `types: { output: {} as {...} }`, a final state or machine-level `output` function, and a note that `actor.getSnapshot().output` exposes it.

17. **Custom actor logic** — Build an actor from scratch via `fromTransition` (or a hand-rolled `ActorLogic`).
    - API: `fromTransition`, reducer-style `(state, event) => nextState`.
    - Must include: `fromTransition((state, event) => ..., initialState)`, an `input` argument when relevant, and invocation from the parent.

18. **Parent-child communication** — Parent invokes a child and they exchange events.
    - API: `invoke` (machine actor), `sendTo`, `sendParent` via `sendTo(({ self }) => self._parent, ...)`.
    - Must include: a child machine defined with its own setup, invoked by id, parent sending it events and child sending events back.

19. **Reentry vs internal transitions** — Distinguish transitions that re-enter a state from ones that stay.
    - API: `reenter: true` (v5 replacement for v4 `internal: false`), default is internal.
    - Must include: two transitions where one uses `reenter: true` to re-fire entry/exit actions, and one stays internal (default), with a comment explaining the difference.

20. **Snapshot persistence** — Serialize and restore machine state.
    - API: `actor.getPersistedSnapshot()`, `createActor(machine, { snapshot })`.
    - Must include: persist via `getPersistedSnapshot()`, rehydrate by passing `snapshot` to `createActor`, and a note that invoked/spawned children are also restored.

21. **Invoked promise with onDone/onError** — Inline promise invocation variant of pattern 3 without predeclaring the actor.
    - API: `invoke: { src: fromPromise(...) }` or referenced src, `onDone`, `onError`.
    - Must include: full `invoke` block with both `onDone` and `onError` transitions and typed access to `event.output` / `event.error`.

22. **Invoked machine** — A state invokes a child machine for its duration.
    - API: `invoke: { src: childMachine, input, onDone }`.
    - Must include: child machine declared with its own `setup()`, registered in `setup.actors`, invoked with typed `input`, and `onDone` consuming the child's typed `output`.

23. **Delayed transitions** — Named delays parameterised from context.
    - API: `after: { DELAY_NAME: ... }`, `setup({ delays })` with `({ context }) => ms`.
    - Must include: a named delay in `setup.delays` that reads `context`, referenced from `after`, plus demonstration of per-instance timing.

24. **Conditional transitions with guards** — Multiple guarded targets for one event.
    - API: `on: { EV: [{ guard: 'a', target: ... }, { guard: 'b', target: ... }, { target: 'fallback' }] }`.
    - Must include: an event with an array of transition objects, guards referenced by name, and a final unguarded fallback entry.

25. **Context updates via assign** — Basic and nested context mutation via `assign`.
    - API: `assign({ key: ({ context, event }) => ... })`, partial updates.
    - Must include: typed context, at least two `assign` calls, one that derives from `context` and one that reads `event`. Show the object form, not the deprecated function form.

26. **Raise events** — Machine raises an internal event to itself.
    - API: `raise({ type: '...' })`, `raise(({ context }) => ({ type, ... }))`.
    - Must include: at least one `raise` inside `entry`, `exit`, or an `actions` array, plus a handler for the raised event in `on`.

27. **History states** — Shallow / deep history for returning to the last active substate.
    - API: `type: 'history'`, `history: 'shallow' | 'deep'`.
    - Must include: a compound state containing a `history` child (shallow by default, explicit `history: 'deep'` when called for), and an external transition that targets the history node.

28. **Error handling in invoked actors** — Surface actor failures cleanly.
    - API: `onError` transitions, typed `event.error`, optional `escalate`.
    - Must include: `onError` that assigns a typed error into context, transition to a dedicated `failed` state, optional retry event path.

29. **Cancel delayed events** — Schedule a delayed send, then cancel it.
    - API: `sendTo` with `{ id: 'foo', delay: N }`, `cancel('foo')`.
    - Must include: a `sendTo`/`raise` scheduled with an `id` and `delay`, and a `cancel(id)` action on a transition that invalidates it (e.g. user navigates away before the timeout).

30. **Referenced actions and guards in setup** — Keep machine body string-referenced; declare implementations once.
    - API: `setup({ actions: { ... }, guards: { ... } })`, string references in states.
    - Must include: at least two actions and two guards declared in `setup`, referenced by string name in transitions / entry / exit, with a note that this gives the best type inference and reuse.
