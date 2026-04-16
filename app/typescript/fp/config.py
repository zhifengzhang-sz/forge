from lib.common.types import TopicConfig, RepoConfig

CONFIG = TopicConfig(
    name="typescript.fp",
    language="typescript",
    repos=[
        RepoConfig(url="https://github.com/gcanti/fp-ts.git", name="fp-ts"),
        RepoConfig(url="https://github.com/Effect-TS/effect.git", name="effect"),
    ],
    file_extensions=[".ts"],
    skip_dirs=["node_modules", "dist", "build", ".git", "__tests__"],
    skip_suffixes=[".d.ts", ".spec.ts", ".test.ts"],
    focus_terms=[
        "pipe(", "pipe,", "flow(", "flow,",
        "Option<", "Option,", "Either<", "Either,",
        "Task<", "Task,", "Reader<", "Reader,",
        "Effect<", "Effect.", "Layer<", "Layer.",
    ],
    scoring_signals=[],   # use TypeScript defaults
    scoring_penalties=[], # use TypeScript defaults
    system_prompt_fragment="functional programming (fp-ts, Effect)",
)

DOMAIN_TERMS = ["fp-ts", "either", "option", "effect-ts"]
