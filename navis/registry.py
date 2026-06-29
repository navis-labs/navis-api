"""Import side-effects registry.

Importing this module ensures:
  1. Every ORM model is registered on `Base.metadata` (so create_all/Alembic see them).
  2. Every event subscriber is imported (so @subscribe handlers attach to the bus).

Both `main.py` and `init_db.py` import this once.
"""
# --- ORM models (table registration) ---
from navis.identity import models as _identity_models  # noqa: F401
from navis.identity import enterprise_models as _identity_enterprise_models  # noqa: F401
from navis.organization import models as _organization_models  # noqa: F401
from navis.signals import models as _signal_models  # noqa: F401
from navis.memory import models as _memory_models  # noqa: F401
from navis.state.v2 import snapshots as _state_v2_snapshots  # noqa: F401
from navis.decisions import models as _decision_models  # noqa: F401
from navis.execution import models as _execution_models  # noqa: F401
from navis.observation import models as _observation_models  # noqa: F401
from navis.learning import models as _learning_models  # noqa: F401
from navis.intelligence import models as _intelligence_models  # noqa: F401
from navis.adaptive import models as _adaptive_models  # noqa: F401
from navis.ml.embeddings import models as _ml_embedding_models  # noqa: F401

# --- Event subscribers + Celery task registration ---
from navis.memory import service as _memory_service  # noqa: F401
from navis.learning import subscribers as _learning_subscribers  # noqa: F401  (OUTCOME_OBSERVED)
from navis.execution import worker as _execution_worker  # noqa: F401  (registers tasks)
from navis.signals import outbox as _signal_outbox  # noqa: F401  (registers the outbox relay task)
from navis.identity import maintenance as _identity_maintenance  # noqa: F401  (registers purge tasks)
from navis.observation import subscribers as _observation_subscribers  # noqa: F401  (closes the loop)
from navis.orchestration import org_provisioning as _org_provisioning  # noqa: F401  (ORG_CREATED + onboarding funnel)
