from lib.common.types import TopicConfig, RepoConfig

CONFIG = TopicConfig(
    name="typescript.eventsourcing",
    language="typescript",
    repos=[
        RepoConfig(url="https://github.com/oskardudycz/EventSourcing.NodeJS.git", name="EventSourcing.NodeJS"),
    ],
    file_extensions=[".ts"],
    skip_dirs=["node_modules", "dist", "build", ".git", "__tests__"],
    skip_suffixes=[".d.ts", ".spec.ts", ".test.ts"],
    focus_terms=[
        "Aggregate", "AggregateStream",
        "evolve(", "evolve<", "evolve:",
        "Command", "CommandHandler",
        "EventStore", "EventStream",
        "append(", "appendToStream(",
        "readStream(", "readFromStream(",
        "Decider", "Projector",
    ],
    scoring_signals=[],
    scoring_penalties=[],
    system_prompt_fragment="event sourcing",
)

DOMAIN_TERMS = ["event sourc", "event store", "aggregate"]
