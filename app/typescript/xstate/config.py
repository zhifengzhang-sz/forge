from lib.common.types import TopicConfig, RepoConfig

CONFIG = TopicConfig(
    name="typescript.xstate",
    language="typescript",
    repos=[
        RepoConfig(url="https://github.com/statelyai/xstate.git", name="xstate"),
    ],
    file_extensions=[".ts"],
    skip_dirs=["node_modules", "dist", "build", ".git", "__tests__"],
    skip_suffixes=[".d.ts", ".spec.ts", ".test.ts"],
    focus_terms=[
        "setup(", "setup<", "createMachine(", "createMachine<",
        "fromPromise(", "fromPromise<", "fromObservable(", "fromObservable<",
        "assign(", "assign<", "StateMachine", "MachineContext",
        "AnyActorRef", "ActorLogic", "EventObject",
    ],
    scoring_signals=[],
    scoring_penalties=[],
    system_prompt_fragment="state machines (XState v5)",
)

DOMAIN_TERMS = ["xstate", "state machine", "statechart"]
