#!/usr/bin/env python3
"""Unit tests for v0.6/verify.py.

Covers parse_code, strip_comments, check_idiom, and check_length.

compile_all is NOT tested here by default: it requires `npm install` into
v0.6/.verify-tmp/, which would be slow (downloads xstate + typescript) and
flaky in sandboxes without network. To opt-in, set RUN_COMPILE_TESTS=1 in
the environment.

Run:
    python v0.6/test_verify.py
    RUN_COMPILE_TESTS=1 python v0.6/test_verify.py
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
        # Only a generic / non-ts fence should also miss.
        self.assertIsNone(parse_code("```\nfoo\n```"))
        self.assertIsNone(parse_code("```python\nprint(1)\n```"))

    def test_concatenates_multiple_fences(self) -> None:
        a = "const a = 1;"
        b = "const b = 2;"
        msg = f"First:\n```ts\n{a}\n```\n\nSecond:\n```typescript\n{b}\n```"
        got = parse_code(msg)
        self.assertIsNotNone(got)
        # Both bodies appear, in order.
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
        # Two separate block comments should not be merged into one span.
        src = "/* a */ keep /* b */"
        got = strip_comments(src)
        self.assertIn("keep", got)
        self.assertNotIn("/*", got)
        self.assertNotIn("*/", got)


# ---------------------------------------------------------------------------
# check_idiom
# ---------------------------------------------------------------------------


class CheckIdiomTests(unittest.TestCase):
    def test_accepts_known_good_v5(self) -> None:
        ok, reason = check_idiom(GOOD_V5)
        self.assertTrue(ok, f"expected pass, got reason={reason!r}")
        self.assertEqual(reason, "")

    def test_rejects_missing_setup(self) -> None:
        # Valid-looking machine but no setup() call.
        code = "import { createMachine } from 'xstate';\nconst m = createMachine({});"
        ok, reason = check_idiom(code)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:missing_setup"), reason)

    def test_rejects_cond_in_code(self) -> None:
        # Real object-literal `cond:` in code → still rejected.
        code = GOOD_V5 + "\nconst opts = { cond: 'isValid' };\n"
        ok, reason = check_idiom(code)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:cond:"), reason)

    def test_accepts_cond_in_line_comment(self) -> None:
        # `cond:` inside a line comment (Phase A prose) should NOT reject.
        code = GOOD_V5 + "\n// In XState v4 you would write { cond: 'isValid' }\n"
        ok, reason = check_idiom(code)
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_accepts_cond_in_block_comment(self) -> None:
        # `cond:` inside a /* */ block comment should NOT reject.
        code = GOOD_V5 + "\n/* v4 migration note: { cond: 'isValid' } */\n"
        ok, reason = check_idiom(code)
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_accepts_cond_in_jsdoc(self) -> None:
        # `cond:` inside a JSDoc comment (common for migration docs).
        code = (
            "import { setup } from 'xstate';\n"
            "/**\n"
            " * Migrated from v4 where guards were written as `cond: 'isValid'`.\n"
            " * In v5, use `guard:` instead.\n"
            " */\n"
            "export const m = setup({}).createMachine({});\n"
        )
        ok, reason = check_idiom(code)
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_setup_only_in_comment_is_rejected(self) -> None:
        # If the only `setup(` is inside a comment, the must-match gate should
        # fail — comment-stripping removes it before checking.
        code = "// use setup(...) here\nimport { createMachine } from 'xstate';\n"
        ok, reason = check_idiom(code)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:missing_setup"), reason)

    def test_rejects_interpret(self) -> None:
        code = GOOD_V5 + "\nconst svc = interpret(counterMachine).start();\n"
        ok, reason = check_idiom(code)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:interpret:"), reason)

    def test_accepts_interpret_in_comment(self) -> None:
        # v4 `interpret(` mentioned in prose should not reject.
        code = GOOD_V5 + "\n// v4 used interpret(machine).start(); v5 uses createActor\n"
        ok, reason = check_idiom(code)
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_rejects_services_block(self) -> None:
        code = GOOD_V5 + "\nconst opts = { services: {\n  fetch: () => {},\n} };\n"
        ok, reason = check_idiom(code)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:services:"), reason)

    def test_rejects_ctx_event_callback(self) -> None:
        code = GOOD_V5 + "\nconst f = (ctx, event) => ctx.count + 1;\n"
        ok, reason = check_idiom(code)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:ctx_event_cb:"), reason)

    def test_rejects_bare_Machine_factory(self) -> None:
        code = "import { Machine } from 'xstate';\nsetup({});\nconst m = Machine({});\n"
        ok, reason = check_idiom(code)
        self.assertFalse(ok)
        self.assertTrue(reason.startswith("idiom:Machine:"), reason)

    def test_does_not_match_second_colon(self) -> None:
        # "second:" (a minute/second event key) should NOT trip the `cond:`
        # regex — there's no word boundary before 'c' in "second:".
        code = GOOD_V5 + "\nconst evt = { second: {} };\n"
        ok, reason = check_idiom(code)
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_does_not_match_microservices_colon(self) -> None:
        # "microservices:" should NOT trip the `services:` regex.
        code = GOOD_V5 + "\nconst x = { microservices: { a: 1 } };\n"
        ok, reason = check_idiom(code)
        self.assertTrue(ok, f"got reason={reason!r}")

    def test_allows_createMachine_and_other_Machine_suffix(self) -> None:
        # `createMachine(` and `toggleMachine(` are legal v5 names — the
        # Machine regex uses a negative lookbehind on letters, so neither
        # should trigger the v4 guard.
        code = (
            "import { setup, createMachine } from 'xstate';\n"
            "const toggleMachine = setup({}).createMachine({});\n"
            "const svc = toggleMachine;\n"
        )
        ok, reason = check_idiom(code)
        self.assertTrue(ok, f"got reason={reason!r}")


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
        # Length gate must count real bytes, including comments — a huge
        # comment-only file is still worth rejecting for verbosity.
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
        cls.tmp = Path(cls._tmp_obj.name)
        setup_probe_dir(cls.tmp)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_obj.cleanup()

    def test_good_code_compiles(self) -> None:
        failures, elapsed, timed_out = compile_all(self.tmp, [(1, GOOD_V5)])
        self.assertFalse(timed_out)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertNotIn(1, failures)

    def test_bad_code_rejected(self) -> None:
        bad = "import { setup } from 'xstate';\nconst x: number = 'not a number';\n"
        failures, _elapsed, timed_out = compile_all(self.tmp, [(2, bad)])
        self.assertFalse(timed_out)
        self.assertIn(2, failures)
        self.assertTrue(failures[2], "expected non-empty tsc detail")

    def test_mixed_batch_attributes_errors_correctly(self) -> None:
        bad = "import { setup } from 'xstate';\nconst x: number = 'not a number';\n"
        failures, _elapsed, timed_out = compile_all(
            self.tmp, [(10, GOOD_V5), (11, bad), (12, GOOD_V5)]
        )
        self.assertFalse(timed_out)
        self.assertNotIn(10, failures)
        self.assertIn(11, failures)
        self.assertNotIn(12, failures)

    def test_cleans_stale_probes(self) -> None:
        # Seed a stale probe that would fail.
        (self.tmp / "probe-999999.ts").write_text("const x: number = 'nope';\n")
        failures, _elapsed, timed_out = compile_all(self.tmp, [(20, GOOD_V5)])
        self.assertFalse(timed_out)
        # 999999 was cleaned up, so no error for it.
        self.assertNotIn(999999, failures)
        self.assertNotIn(20, failures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
