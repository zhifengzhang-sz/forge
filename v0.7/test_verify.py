#!/usr/bin/env python3
"""Unit tests for v0.7/verify.py.

Covers parse_code, strip_comments, check_idiom (per-domain), and check_length.

compile_all is NOT tested here by default: it requires `npm install` into
per-domain probe dirs, which would be slow (downloads xstate + typescript +
fp-ts + effect + rxjs) and flaky in sandboxes without network. To opt-in,
set RUN_COMPILE_TESTS=1 in the environment.

Run:
    python v0.7/test_verify.py
    RUN_COMPILE_TESTS=1 python v0.7/test_verify.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verify import (  # noqa: E402
    check_idiom,
    check_length,
    compile_all,
    parse_code,
    setup_probe_dir,
    strip_comments,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOOD_V5 = """\
import { setup, assign } from 'xstate';

export const counterMachine = setup({
  types: {
    context: {} as { count: number },
    events: {} as { type: 'INC' },
  },
}).createMachine({
  id: 'counter',
  context: { count: 0 },
  on: {
    INC: { actions: assign({ count: ({ context }) => context.count + 1 }) },
  },
});
"""

GOOD_FP_EFFECT_GEN = """\
import { Effect, pipe } from 'effect';

interface User { readonly id: string; readonly name: string }

const getUser = (id: string): Effect.Effect<User, Error> =>
  Effect.succeed({ id, name: 'Ada' });

const program = Effect.gen(function* () {
  const u = yield* getUser('42');
  yield* Effect.log(`hello ${u.name}`);
  return u;
});

const run = pipe(program, Effect.tap((u) => Effect.log(u.id)));
"""

GOOD_FP_FP_TS_EITHER = """\
import * as E from 'fp-ts/Either';
import { pipe } from 'fp-ts/function';

type Err = { readonly _tag: 'ParseError'; readonly msg: string };

const parseNumber = (s: string): E.Either<Err, number> => {
  const n = Number(s);
  return Number.isFinite(n)
    ? E.right(n)
    : E.left({ _tag: 'ParseError' as const, msg: s });
};

const doubled = pipe(parseNumber('7'), E.chain((n) => E.right(n * 2)));
"""

GOOD_RX_SWITCHMAP = """\
import { of, Observable } from 'rxjs';
import { switchMap, tap } from 'rxjs/operators';

const source$: Observable<number> = of(1, 2, 3);

const stream$ = source$.pipe(
  switchMap((n) => of(n * 2)),
  tap((n) => console.log(n)),
);
"""

GOOD_RX_COMBINELATEST = """\
import { BehaviorSubject, combineLatest } from 'rxjs';
import { map } from 'rxjs/operators';

const first$ = new BehaviorSubject<number>(1);
const second$ = new BehaviorSubject<number>(2);

const sum$ = combineLatest([first$, second$]).pipe(
  map(([a, b]) => a + b),
);
"""

GOOD_ES_DECIDER = """\
// Decider-style event-sourcing: pure evolve + decide, no I/O.

type State =
  | { status: 'Empty' }
  | { status: 'Active'; balance: number };

type Command =
  | { type: 'Open'; opening: number }
  | { type: 'Credit'; amount: number };

type Event =
  | { type: 'Opened'; balance: number }
  | { type: 'Credited'; amount: number };

export const initial: State = { status: 'Empty' };

export function evolve(state: State, event: Event): State {
  switch (event.type) {
    case 'Opened':
      return { status: 'Active', balance: event.balance };
    case 'Credited':
      if (state.status !== 'Active') return state;
      return { status: 'Active', balance: state.balance + event.amount };
    default: {
      const _: never = event;
      return state;
    }
  }
}

export function decide(cmd: Command, state: State): Event[] {
  switch (cmd.type) {
    case 'Open':
      if (state.status === 'Active') throw new Error('AlreadyOpen');
      return [{ type: 'Opened', balance: cmd.opening }];
    case 'Credit':
      if (state.status !== 'Active') throw new Error('NotOpen');
      return [{ type: 'Credited', amount: cmd.amount }];
    default: {
      const _: never = cmd;
      return [];
    }
  }
}
"""

GOOD_ES_OSKAR_WHEN = """\
// Oskar-style: pure `when` reducer + generic aggregateStream helper.

interface DomainEvent {
  type: string;
}

interface Invoice {
  id: string;
  issued: boolean;
  totalCents: number;
}

type InvoiceEvent =
  | { type: 'Issued'; id: string }
  | { type: 'LineAdded'; cents: number };

export function when(state: Partial<Invoice>, event: InvoiceEvent): Partial<Invoice> {
  switch (event.type) {
    case 'Issued':
      return { ...state, id: event.id, issued: true, totalCents: 0 };
    case 'LineAdded':
      return { ...state, totalCents: (state.totalCents ?? 0) + event.cents };
    default:
      return state;
  }
}

export async function aggregateStream<Aggregate, E extends DomainEvent>(
  stream: AsyncIterable<E>,
  evolve: (s: Partial<Aggregate>, e: E) => Partial<Aggregate>,
  initial: Partial<Aggregate> = {},
): Promise<Partial<Aggregate>> {
  let state = initial;
  for await (const event of stream) {
    state = evolve(state, event);
  }
  return state;
}
"""

GOOD_ES_EFFECT_WRAPPED = """\
// Effect-wrapped event-store access wrapping a pure Decider core.

import { Effect } from 'effect';

interface EventStore<E> {
  readStream(name: string): Promise<E[]>;
  appendToStream(name: string, events: E[], expectedRevision: bigint | 'no_stream'): Promise<bigint>;
}

type Event = { type: 'Opened' } | { type: 'Closed' };

type State = { open: boolean };

const initial: State = { open: false };

const evolve = (s: State, e: Event): State =>
  e.type === 'Opened' ? { open: true } : { open: false };

const loadAggregate = <E, S>(
  store: EventStore<E>,
  streamName: string,
  initialState: S,
  step: (s: S, e: E) => S,
): Effect.Effect<S, Error> =>
  Effect.tryPromise({
    try: () => store.readStream(streamName),
    catch: (u) => new Error(String(u)),
  }).pipe(Effect.map((events) => events.reduce(step, initialState)));

export const load = (store: EventStore<Event>, name: string) =>
  loadAggregate(store, name, initial, evolve);
"""

BAD_ES_XSTATE_LEAKAGE = """\
// Has an ES positive token (evolve) but imports from xstate — must fail.
import { setup } from 'xstate';

export function evolve(state: unknown): unknown {
  return state;
}
"""

BAD_ES_SETUP_LEAKAGE = """\
// Has an ES positive token (Decider) but uses setup() — must fail.
interface Decider<C, E, S> {}

const m = setup({ types: {} }).createMachine({ id: 'x' });

export {};
"""


def _wrap_ts(body: str, tag: str = "typescript") -> str:
    return f"Here you go:\n\n```{tag}\n{body}\n```\n"


# ---------------------------------------------------------------------------
# parse_code
# ---------------------------------------------------------------------------


class ParseCodeTests(unittest.TestCase):
    def test_returns_body_for_typescript_fence(self) -> None:
        wrapped = _wrap_ts(GOOD_V5, "typescript")
        got = parse_code(wrapped)
        self.assertIsNotNone(got)
        self.assertIn("setup(", got)
        self.assertIn("counterMachine", got)

    def test_returns_body_for_ts_fence(self) -> None:
        wrapped = _wrap_ts(GOOD_V5, "ts")
        got = parse_code(wrapped)
        self.assertIsNotNone(got)
        self.assertIn("setup(", got)

    def test_returns_none_when_no_fence(self) -> None:
        self.assertIsNone(parse_code("no code here"))
        self.assertIsNone(parse_code(""))
        self.assertIsNone(parse_code("```\nfoo\n```"))
        self.assertIsNone(parse_code("```python\nprint(1)\n```"))

    def test_concatenates_multiple_fences(self) -> None:
        a = "const a = 1;"
        b = "const b = 2;"
        msg = f"First:\n```ts\n{a}\n```\n\nSecond:\n```typescript\n{b}\n```"
        got = parse_code(msg)
        self.assertIsNotNone(got)
        self.assertIn(a, got)
        self.assertIn(b, got)
        self.assertLess(got.index(a), got.index(b))

    def test_handles_non_string_input(self) -> None:
        self.assertIsNone(parse_code(None))  # type: ignore[arg-type]
        self.assertIsNone(parse_code(123))   # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# strip_comments
# ---------------------------------------------------------------------------


class StripCommentsTests(unittest.TestCase):
    def test_removes_line_comments(self) -> None:
        got = strip_comments("const x = 1; // trailing\nconst y = 2;")
        self.assertNotIn("trailing", got)
        self.assertIn("const x = 1;", got)
        self.assertIn("const y = 2;", got)

    def test_removes_block_comments(self) -> None:
        got = strip_comments("/* header */ const x = 1; /* inline */ const y = 2;")
        self.assertNotIn("header", got)
        self.assertNotIn("inline", got)
        self.assertIn("const x = 1;", got)
        self.assertIn("const y = 2;", got)

    def test_removes_multiline_block_comments(self) -> None:
        src = "/* multi\n * line\n * block\n */\nconst x = 1;"
        got = strip_comments(src)
        self.assertNotIn("multi", got)
        self.assertNotIn("block", got)
        self.assertIn("const x = 1;", got)

    def test_non_greedy_block_comments(self) -> None:
        src = "/* a */ keep /* b */"
        got = strip_comments(src)
        self.assertIn("keep", got)
        self.assertNotIn("/*", got)
        self.assertNotIn("*/", got)


# ---------------------------------------------------------------------------
# check_idiom — xstate (v0.6 parity)
# ---------------------------------------------------------------------------


class CheckIdiomXStateTests(unittest.TestCase):
    def test_accepts_known_good_v5(self) -> None:
        ok, reason = check_idiom(GOOD_V5, domain="xstate")
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")
        self.assertEqual(reason, "")

    def test_default_domain_is_xstate(self) -> None:
        # Calling without a domain kwarg should behave like v0.6 (xstate).
        ok, reason = check_idiom(GOOD_V5)
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_rejects_missing_setup(self) -> None:
        code = "import { createMachine } from 'xstate';\nconst m = createMachine({});"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:missing_setup"), reason)

    def test_rejects_cond_in_code(self) -> None:
        code = GOOD_V5 + "\nconst opts = { cond: 'isValid' };\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:cond:"), reason)

    def test_accepts_cond_in_line_comment(self) -> None:
        code = GOOD_V5 + "\n// In XState v4 you would write { cond: 'isValid' }\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_accepts_cond_in_block_comment(self) -> None:
        code = GOOD_V5 + "\n/* v4 migration note: { cond: 'isValid' } */\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_accepts_cond_in_jsdoc(self) -> None:
        code = (
            "import { setup } from 'xstate';\n"
            "/**\n"
            " * Migrated from v4 where guards were written as `cond: 'isValid'`.\n"
            " * In v5, use `guard:` instead.\n"
            " */\n"
            "export const m = setup({}).createMachine({});\n"
        )
        ok, reason = check_idiom(code, domain="xstate")
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_setup_only_in_comment_is_rejected(self) -> None:
        code = "// use setup(...) here\nimport { createMachine } from 'xstate';\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:missing_setup"), reason)

    def test_rejects_interpret(self) -> None:
        code = GOOD_V5 + "\nconst svc = interpret(counterMachine).start();\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:interpret:"), reason)

    def test_accepts_interpret_in_comment(self) -> None:
        code = GOOD_V5 + "\n// v4 used interpret(machine).start(); v5 uses createActor\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_rejects_services_block(self) -> None:
        code = GOOD_V5 + "\nconst opts = { services: {\n  fetch: () => {},\n} };\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:services:"), reason)

    def test_rejects_ctx_event_callback(self) -> None:
        code = GOOD_V5 + "\nconst f = (ctx, event) => ctx.count + 1;\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:ctx_event_cb:"), reason)

    def test_rejects_bare_Machine_factory(self) -> None:
        code = "import { Machine } from 'xstate';\nsetup({});\nconst m = Machine({});\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:Machine:"), reason)

    def test_does_not_match_second_colon(self) -> None:
        # "second:" (a minute/second event key) should NOT trip the `cond:`
        # regex — there's no word boundary before 'c' in "second:".
        code = GOOD_V5 + "\nconst evt = { second: {} };\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_does_not_match_microservices_colon(self) -> None:
        code = GOOD_V5 + "\nconst x = { microservices: { a: 1 } };\n"
        ok, reason = check_idiom(code, domain="xstate")
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_allows_createMachine_and_other_Machine_suffix(self) -> None:
        code = (
            "import { setup, createMachine } from 'xstate';\n"
            "const toggleMachine = setup({}).createMachine({});\n"
            "const svc = toggleMachine;\n"
        )
        ok, reason = check_idiom(code, domain="xstate")
        self.assertTrue(ok, f"got reason={reason!r}")


# ---------------------------------------------------------------------------
# check_idiom — fp (new)
# ---------------------------------------------------------------------------


class CheckIdiomFPTests(unittest.TestCase):
    def test_accepts_effect_gen(self) -> None:
        ok, reason = check_idiom(GOOD_FP_EFFECT_GEN, domain="fp")
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")

    def test_accepts_fp_ts_either_chain(self) -> None:
        ok, reason = check_idiom(GOOD_FP_FP_TS_EITHER, domain="fp")
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")

    def test_rejects_xstate_only_code_in_fp(self) -> None:
        # Code that only uses XState imports should fail the fp positive
        # gate (no Effect/fp-ts tokens) AND/or the xstate-import shape guard.
        # Check that we get *some* idiom rejection.
        code = "import { createMachine } from 'xstate';\nconst m = createMachine({});\n"
        ok, reason = check_idiom(code, domain="fp")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:"), reason)

    def test_rejects_setup_shape_in_fp(self) -> None:
        # A file with an `effect` import but containing setup({...}) shape —
        # shape guard should fire. The MUST-MATCH passes (Effect.succeed
        # present) so the rejection reason is the setup_in_fp shape guard.
        code = (
            "import { Effect } from 'effect';\n"
            "const e = Effect.succeed(1);\n"
            "// Someone pasted XState code by mistake:\n"
            'const m = setup({ types: {} }).createMachine({ id: "x" });\n'
        )
        ok, reason = check_idiom(code, domain="fp")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:setup_in_fp"), reason)

    def test_rejects_xstate_import_in_fp(self) -> None:
        # fp positive token present, but file imports from xstate → reject
        # with the xstate-import-in-fp shape guard.
        code = (
            "import { Effect } from 'effect';\n"
            "import { createMachine } from 'xstate';\n"
            "const e = Effect.succeed(createMachine({}));\n"
        )
        ok, reason = check_idiom(code, domain="fp")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:xstate_import_in_fp"), reason)

    def test_rejects_effect_service_deprecated(self) -> None:
        code = (
            "import { Effect } from 'effect';\n"
            "class Foo {}\n"
            "const svc = Effect.service(Foo);\n"
            "const e = Effect.succeed(1);\n"
        )
        ok, reason = check_idiom(code, domain="fp")
        self.assertFalse(ok)
        self.assertTrue(
            reason.startswith("idiom:Effect_service_deprecated"), reason
        )

    def test_accepts_pipe_only_fp(self) -> None:
        # pipe( alone is enough for the fp positive gate.
        code = (
            "import { pipe } from 'fp-ts/function';\n"
            "const add1 = (n: number): number => n + 1;\n"
            "export const r = pipe(1, add1, add1);\n"
        )
        ok, reason = check_idiom(code, domain="fp")
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_rejects_fp_missing_positive(self) -> None:
        # Plain TS with no fp-ts / effect tokens and no xstate leakage —
        # should fail the missing_fp_positive gate.
        code = (
            "import { readFileSync } from 'fs';\n"
            "const contents = readFileSync('x.txt', 'utf8');\n"
            "const lines = contents.split('\\n');\n"
        )
        ok, reason = check_idiom(code, domain="fp")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:missing_fp_positive"), reason)

    def test_accepts_effect_in_comment_still_needs_real_token(self) -> None:
        # If Effect.gen only appears in a comment, positive gate should NOT
        # see it (comment-strip parity with v0.6).
        code = (
            "import { readFileSync } from 'fs';\n"
            "// use Effect.gen here in a refactor\n"
            "const x = readFileSync('f', 'utf8');\n"
        )
        ok, reason = check_idiom(code, domain="fp")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:missing_fp_positive"), reason)


# ---------------------------------------------------------------------------
# check_idiom — rx (new)
# ---------------------------------------------------------------------------


class CheckIdiomRXTests(unittest.TestCase):
    def test_accepts_of_switchmap(self) -> None:
        ok, reason = check_idiom(GOOD_RX_SWITCHMAP, domain="rx")
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")

    def test_accepts_combinelatest(self) -> None:
        ok, reason = check_idiom(GOOD_RX_COMBINELATEST, domain="rx")
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")

    def test_rejects_map_only_no_pipe_no_other_operators(self) -> None:
        # Code that uses only `map` (Array method) with no pipe() and no
        # other RxJS operators → fails the rx positive gate. This is the
        # "v7 requirement" check: pre-v6 RxJS did `.map()` directly on
        # observables; v7 requires pipe().
        code = (
            "const arr = [1, 2, 3];\n"
            "const doubled = arr.map((n) => n * 2);\n"
            "console.log(doubled);\n"
        )
        ok, reason = check_idiom(code, domain="rx")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:missing_rx_positive"), reason)

    def test_rejects_xstate_import_in_rx(self) -> None:
        # RxJS positive token present, but the file imports from xstate →
        # reject with the xstate-import-in-rx shape guard.
        code = (
            "import { Observable } from 'rxjs';\n"
            "import { createMachine } from 'xstate';\n"
            "const obs: Observable<number> = new Observable((s) => s.next(1));\n"
        )
        ok, reason = check_idiom(code, domain="rx")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:xstate_import_in_rx"), reason)

    def test_rejects_setup_in_rx(self) -> None:
        # File imports from rxjs but has setup({...}) shape — shape guard.
        code = (
            "import { Observable } from 'rxjs';\n"
            "const o: Observable<number> = new Observable((s) => s.next(1));\n"
            'const m = setup({ types: {} }).createMachine({ id: "x" });\n'
        )
        ok, reason = check_idiom(code, domain="rx")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:setup_in_rx"), reason)

    def test_rejects_pre_v7_do_operator(self) -> None:
        code = (
            "import { of } from 'rxjs';\n"
            "const s = of(1, 2, 3).do((n) => console.log(n));\n"
        )
        ok, reason = check_idiom(code, domain="rx")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:rxjs_do_operator"), reason)

    def test_accepts_behaviorsubject_alone(self) -> None:
        # BehaviorSubject is in the positive list; a small record using it
        # should pass.
        code = (
            "import { BehaviorSubject } from 'rxjs';\n"
            "const count$ = new BehaviorSubject<number>(0);\n"
            "count$.subscribe((n) => console.log(n));\n"
        )
        ok, reason = check_idiom(code, domain="rx")
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_accepts_rx_in_jsdoc_cond_comment(self) -> None:
        # A rx-positive file that mentions v4 `.do()` in a comment should
        # NOT be rejected by the do-operator guard (comment-stripped).
        code = (
            "import { of, tap } from 'rxjs';\n"
            "// Pre-v7 we used .do() — in v7 use tap() instead\n"
            "const s = of(1).pipe(tap((n) => console.log(n)));\n"
        )
        ok, reason = check_idiom(code, domain="rx")
        self.assertTrue(ok, f"got reason={reason!r}")


# ---------------------------------------------------------------------------
# check_idiom — es (new in v1)
# ---------------------------------------------------------------------------


class CheckIdiomESTests(unittest.TestCase):
    def test_accepts_decider_style(self) -> None:
        ok, reason = check_idiom(GOOD_ES_DECIDER, domain="es")
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")

    def test_accepts_oskar_when_style(self) -> None:
        ok, reason = check_idiom(GOOD_ES_OSKAR_WHEN, domain="es")
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")

    def test_accepts_effect_wrapped_es(self) -> None:
        # Effect.gen + ES is a valid idiom — ES gate does NOT block Effect.
        ok, reason = check_idiom(GOOD_ES_EFFECT_WRAPPED, domain="es")
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")

    def test_rejects_xstate_import_in_es(self) -> None:
        ok, reason = check_idiom(BAD_ES_XSTATE_LEAKAGE, domain="es")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:xstate_import_in_es"), reason)

    def test_rejects_setup_in_es(self) -> None:
        ok, reason = check_idiom(BAD_ES_SETUP_LEAKAGE, domain="es")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:setup_in_es"), reason)

    def test_rejects_es_missing_positive(self) -> None:
        # Plain TS with no ES vocabulary and no XState leakage → must fail
        # the missing_es_positive gate.
        code = (
            "const xs: number[] = [1, 2, 3];\n"
            "const total = xs.reduce((a, b) => a + b, 0);\n"
            "console.log(total);\n"
        )
        ok, reason = check_idiom(code, domain="es")
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:missing_es_positive"), reason)

    def test_accepts_es_in_jsdoc_setup_is_stripped(self) -> None:
        # Comment mentioning setup() must not trip the setup_in_es guard.
        code = (
            "// Migrating away from `setup({...}).createMachine(...)` XState.\n"
            "export const foldEvents = (events: unknown[]): unknown =>\n"
            "  events.reduce((s, e) => evolve(s, e as never), {});\n"
            "function evolve(state: unknown, _e: unknown): unknown { return state; }\n"
        )
        # Positive gate: `evolve(` appears twice (call-site + def). Shape
        # guard: `setup(` only in a comment, which strip_comments removes
        # before the negative gate runs.
        ok, reason = check_idiom(code, domain="es")
        self.assertTrue(ok, f"got reason={reason!r}")


# ---------------------------------------------------------------------------
# check_idiom — domain validation
# ---------------------------------------------------------------------------


class CheckIdiomDomainValidationTests(unittest.TestCase):
    def test_unknown_domain_raises(self) -> None:
        with self.assertRaises(ValueError):
            check_idiom(GOOD_V5, domain="nonsense")


# ---------------------------------------------------------------------------
# check_length
# ---------------------------------------------------------------------------


class CheckLengthTests(unittest.TestCase):
    def test_accepts_midrange(self) -> None:
        code = "x" * 1000
        ok, reason = check_length(code)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_accepts_boundaries(self) -> None:
        self.assertTrue(check_length("x" * 150)[0])
        self.assertTrue(check_length("x" * 6000)[0])

    def test_rejects_too_short(self) -> None:
        ok, reason = check_length("x" * 149)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("length:short:"), reason)

    def test_rejects_too_long(self) -> None:
        ok, reason = check_length("x" * 6001)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("length:long:"), reason)

    def test_length_counts_comments(self) -> None:
        code = "/* " + ("x" * 6000) + " */\nsetup();"
        ok, _reason = check_length(code)
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# compile_all (opt-in; skipped by default)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    os.environ.get("RUN_COMPILE_TESTS") == "1",
    "compile_all tests require npm install; set RUN_COMPILE_TESTS=1 to opt in",
)
class CompileAllTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_obj = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp_obj.name)
        cls.xstate_dir = setup_probe_dir(cls.root, "xstate")
        cls.fp_dir = setup_probe_dir(cls.root, "fp")
        cls.rx_dir = setup_probe_dir(cls.root, "rx")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_obj.cleanup()

    def test_good_xstate_compiles(self) -> None:
        failures, elapsed, timed_out = compile_all(self.xstate_dir, [(1, GOOD_V5)])
        self.assertFalse(timed_out)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertNotIn(1, failures)

    def test_bad_code_rejected(self) -> None:
        bad = "import { setup } from 'xstate';\nconst x: number = 'not a number';\n"
        failures, _elapsed, timed_out = compile_all(self.xstate_dir, [(2, bad)])
        self.assertFalse(timed_out)
        self.assertIn(2, failures)
        self.assertTrue(failures[2], "expected non-empty tsc detail")

    def test_good_fp_compiles(self) -> None:
        failures, _elapsed, timed_out = compile_all(
            self.fp_dir, [(3, GOOD_FP_EFFECT_GEN)]
        )
        self.assertFalse(timed_out)
        self.assertNotIn(3, failures)

    def test_good_rx_compiles(self) -> None:
        failures, _elapsed, timed_out = compile_all(
            self.rx_dir, [(4, GOOD_RX_SWITCHMAP)]
        )
        self.assertFalse(timed_out)
        self.assertNotIn(4, failures)

    def test_mixed_batch_attributes_errors_correctly(self) -> None:
        bad = "import { setup } from 'xstate';\nconst x: number = 'not a number';\n"
        failures, _elapsed, timed_out = compile_all(
            self.xstate_dir, [(10, GOOD_V5), (11, bad), (12, GOOD_V5)]
        )
        self.assertFalse(timed_out)
        self.assertNotIn(10, failures)
        self.assertIn(11, failures)
        self.assertNotIn(12, failures)

    def test_cleans_stale_probes(self) -> None:
        (self.xstate_dir / "probe-999999.ts").write_text(
            "const x: number = 'nope';\n"
        )
        failures, _elapsed, timed_out = compile_all(
            self.xstate_dir, [(20, GOOD_V5)]
        )
        self.assertFalse(timed_out)
        self.assertNotIn(999999, failures)
        self.assertNotIn(20, failures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
