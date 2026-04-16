from lib.common.types import TopicConfig, RepoConfig

CONFIG = TopicConfig(
    name="typescript.reactive",
    language="typescript",
    repos=[
        RepoConfig(url="https://github.com/ReactiveX/rxjs.git", name="rxjs"),
    ],
    file_extensions=[".ts"],
    skip_dirs=["node_modules", "dist", "build", ".git", "__tests__"],
    skip_suffixes=[".d.ts", ".spec.ts", ".test.ts"],
    focus_terms=["Observable<", "Subject", "switchMap(", "mergeMap(", "combineLatest"],
    scoring_signals=[],
    scoring_penalties=[],
    system_prompt_fragment="reactive programming (RxJS)",
)

DOMAIN_TERMS = ["rxjs", "observable", "reactive stream"]
