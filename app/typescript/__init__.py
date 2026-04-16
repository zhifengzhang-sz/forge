"""TypeScript topic configurations."""

from app.typescript.fp.config import CONFIG as FP_CONFIG
from app.typescript.reactive.config import CONFIG as REACTIVE_CONFIG
from app.typescript.xstate.config import CONFIG as XSTATE_CONFIG
from app.typescript.eventsourcing.config import CONFIG as EVENTSOURCING_CONFIG

ALL_TOPICS = [FP_CONFIG, REACTIVE_CONFIG, XSTATE_CONFIG, EVENTSOURCING_CONFIG]
