# Instruction Phrasing Templates

Ten phrasing templates used by Phase A synthesis. A subagent picks one pattern
from `patterns.md`, one phrasing template below, fills the slots, and asks the
teacher to produce a v5 answer. Varied phrasings are the reason ~30 patterns
can explode to ~1500 candidates while still feeling like real developer
prompts.

Slot conventions:

- `{pattern}` — a pattern name from `patterns.md` (e.g. "fetch machine", "multi-step form").
- `{capability}` — a single v5 capability add-on (e.g. "a retry button", "an error state", "a cancel action").
- `{snippet}` — a short prior code block (v4 code, a half-written machine, an XState machine missing a feature).
- `{goal}` — a short natural-language outcome (e.g. "debounce an event for 500 ms", "persist machine state to localStorage").
- `{v4_construct}` — a v4 idiom (e.g. `cond`, `services`, `Machine(...)`, `interpret`, positional action signatures).
- `{constraint}` — an additional requirement (e.g. "strict types, no `any`", "actions declared in setup", "use `enqueueActions`").

---

1. **Build-from-scratch**
   `Build a {pattern} in XState v5 using setup() with typed events and context.`
   *When to use:* default greenfield prompt. Produces the cleanest training
   signal — the completion is a full self-contained machine.

2. **v4 → v5 migration**
   `Convert this XState v4 machine to v5 using setup().createMachine: {snippet}`
   *When to use:* when `{snippet}` contains recognisable v4 idioms (`Machine(...)`,
   `cond`, `services`, positional `(ctx, event) =>` signatures). Trains the
   model on the exact delta it keeps getting wrong on the v0 eval.

3. **Targeted fix**
   `Fix this XState v4 code to v5 idioms: {snippet}`
   *When to use:* shorter than the migration template — used when the snippet is
   only a fragment (a single transition, a single guard). Teaches local
   corrections without requiring a whole new machine.

4. **Capability add-on**
   `Add {capability} to this v5 machine: {snippet}`
   *When to use:* when `{snippet}` is already valid v5 but missing a specific
   feature. Forces the model to read existing typed context/events and extend
   them consistently.

5. **How-to question**
   `What's the v5 way to {goal}? Show a minimal example.`
   *When to use:* for short, conceptual prompts where the answer is 10–30 lines.
   Produces pedagogical snippets with a brief explanation.

6. **v4 → v5 equivalence lookup**
   `Show me the v5 equivalent of {v4_construct}. Brief example.`
   *When to use:* tightly-scoped teaching prompt. Good for producing paired
   examples (v4 form shown in prose, v5 form in code).

7. **Strict-types variant**
   `Implement {pattern} in XState v5 with strict types — no \`any\`, all events and context typed via setup.types.`
   *When to use:* raise the bar beyond template 1 when `{pattern}` benefits from
   showing discriminated unions, `ActorRefFrom`, typed actor outputs, etc.

8. **Refactor for idiom**
   `Refactor this v5 machine to use {constraint}: {snippet}`
   *When to use:* `{snippet}` is working v5 but inline-heavy. Drives the model
   toward `setup({ actions, guards, actors })` references and
   `enqueueActions` where appropriate.

9. **Explain-then-implement**
   `Explain briefly how {pattern} works in XState v5, then show a working typed implementation.`
   *When to use:* produces completions with a short prose prefix before the
   code block. Trains the voice/tone closer to the Claude Opus baseline (see
   `v0/results/claude_opus.json`).

10. **Constraint-driven spec**
    `Write an XState v5 machine that {goal}. Constraints: {constraint}.`
    *When to use:* open-ended specification prompt. The synthesiser should pick
    1–3 concrete constraints (strict types, no inline actions, named delays
    only, etc.) to vary phrasing and force the model to honour them.
