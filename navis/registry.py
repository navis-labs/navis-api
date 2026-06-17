"""Import side-effects registry.

Importing this module ensures:
  1. Every ORM model is registered on `Base.metadata` (so create_all/Alembic see them).
  2. Every event subscriber is imported (so @subscribe handlers attach to the bus).

Both `main.py` and `init_db.py` import this once.
"""
# --- ORM models (table registration) ---
from navis.identity import models as _identity_models  # noqa: F401
from navis.signals import models as _signal_models  # noqa: F401
from navis.memory import models as _memory_models  # noqa: F401
from navis.state import models as _state_models  # noqa: F401  (legacy snapshot table)
from navis.state.v2 import snapshots as _state_v2_snapshots  # noqa: F401
from navis.decisions import models as _decision_models  # noqa: F401
from navis.execution import models as _execution_models  # noqa: F401
from navis.observation import models as _observation_models  # noqa: F401
from navis.learning import models as _learning_models  # noqa: F401

# --- Event subscribers (handler registration) ---
from navis.memory import service as _memory_service  # noqa: F401
from navis.learning import service as _learning_service  # noqa: F401
from navis.orchestration import wiring as _wiring  # noqa: F401
