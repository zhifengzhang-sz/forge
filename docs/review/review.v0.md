```blockquote
The one thing not resolved by v0: review finding #4's deeper claim — that any extraction-based
approach is wrong because users want usage not internals. v0 sidesteps this by hand-curating instead of 
extracting; outcome (C) implicitly tests the extraction premise.
```

---


You're right — that's a real gap and worth being explicit about. Let me think it through.

## What v0 actually tests vs. what finding #4 claims

Finding #4 was two claims bundled together:

1. **Weak claim:** your current extraction is mis-targeted — OSS library *internals* are the wrong source because users ask for *usage*.
2. **Strong claim:** any extraction-based approach is the wrong framing; hand-curation or different data *categories* are required.

v0 tests #1 implicitly — if (C) happens with hand-curated usage examples, then either extraction was wrong OR fine-tuning was wrong; you can't tell which. And v0 doesn't touch #2 at all, because it doesn't include any extraction variant.

So outcome (C) is actually ambiguous in a way I glossed over. "Hand-curation didn't beat base" could mean:

- The base model is already fine at this (premise wrong)
- 50 examples too few (quantity problem)
- Fine-tuning can't move this base model much on this task (approach wrong)
- Hand-curation quality wasn't as good as I assumed (execution problem)

None of those tell you whether extraction from the right sources would have worked.

## Fix: add a cheap extraction arm to v0

The fix is small. Keep v0 as planned, but before Phase 2, spend 30 minutes producing a **second** tiny dataset from extraction — specifically targeting *usage* not internals — and train a second adapter from it. Compare three bars, not two:

| Variant | Source | ~Effort | What it isolates |
|---|---|---|---|
| Base | — | 0 | Does the base need help at all? |
| v0-curated | 50 hand-written usage examples | 2-3 hrs | Does fine-tuning work when data is ideal? |
| v0-extracted | 50 extracted usage examples | 30 min | Does extraction work when source is right? |

For the extracted arm, you don't rebuild your pipeline. You do something deliberately crude:

```bash
# Target usage, not internals. XState's own examples/ directory,
# plus a downstream consumer or two.
git clone --depth=1 https://github.com/statelyai/xstate
git clone --depth=1 https://github.com/kentcdodds/xstate-tutorial  # or similar consumer repo

# Pull every file under examples/ and src/ from tutorial repos.
# Skip the xstate library's own src/ entirely — that's internals.
# Wrap each file or each exported machine in a generic instruction template.
```

The point isn't to build a good extractor. It's to get 50 examples of the *shape of data you think extraction should produce if retargeted correctly*, cheaply enough that you can afford the comparison.

## What the 3-way comparison tells you

| Base | Curated | Extracted | Interpretation |
|---|---|---|---|
| Low | High | High | Extraction works if sources are right. Retarget your pipeline to usage sources. The existing infrastructure is mostly salvageable. |
| Low | High | Low | Finding #4 strong claim is right. Hand-curation is the way; retire the pipeline. |
| Low | Low | Low | Fine-tuning approach is wrong, not just the data. Reconsider the premise (RAG? prompting? different base?). |
| High | — | — | Base is fine. Ship it. |

The fourth row — where extracted beats curated — is genuinely possible and interesting if it happens, because it would mean *scale* (even of imperfect data) beats *quality* (of hand-written data) at this data range, which would argue for doubling down on extraction, not retiring it.

## The honest cost

This adds maybe 90 minutes to v0, mostly one extra training run and grading 20 more outputs. It turns outcome (C) from ambiguous into diagnostic. Worth it.

One thing I'd flag: make the extracted-arm's *instructions* match the curated arm's style. Don't let your template generator's phrasing biases contaminate the comparison. Easiest way: write one instruction shape by hand, apply it to all 50 extracted completions mechanically. You're testing the data source, not the instruction generator.

## What this still doesn't test

Honesty check: even the 3-way comparison doesn't settle whether 500 or 5,000 examples would change the picture, and doesn't test extraction+curation together (which is probably what v1 actually is if curated wins). But those are scale questions appropriate for v1; v0's job is to narrow the search space, not close it.

If curated wins in v0, v1 is "scale curation, possibly with synthetic augmentation." If extracted wins, v1 is "retarget the existing pipeline to usage sources." If both win roughly equally, v1 is "combine them, weight by what's cheaper to produce." Very different projects.