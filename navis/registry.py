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
from navis.identity_resolution import models as _identity_resolution_models  # noqa: F401
from navis.state.v2 import snapshots as _state_v2_snapshots  # noqa: F401
from navis.reasoning import store as _reasoning_store  # noqa: F401  (reasoning_signals table)
from navis.decisions import models as _decision_models  # noqa: F401
from navis.execution import models as _execution_models  # noqa: F401
from navis.execution.drafts import models as _draft_models  # noqa: F401
from navis.observation import models as _observation_models  # noqa: F401
from navis.learning import models as _learning_models  # noqa: F401
from navis.intelligence import models as _intelligence_models  # noqa: F401
from navis.adaptive import models as _adaptive_models  # noqa: F401
from navis.ml.embeddings import models as _ml_embedding_models  # noqa: F401
# Agent execution lifecycle (0048): Task -> Run -> Step -> Outcome. A VIEW over
# the existing machine identity — `agent_id` is a `service_accounts.id` — not a
# second identity model. Registered here so Alembic sees the four tables.
from navis.developer import run_models as _developer_run_models  # noqa: F401

# --- Event subscribers + Celery task registration ---
from navis.memory import service as _memory_service  # noqa: F401
from navis.identity_resolution import service as _identity_resolution_service  # noqa: F401  (subscribes to signal pointers)
from navis.learning import subscribers as _learning_subscribers  # noqa: F401  (OUTCOME_OBSERVED)
# Without this import the LEARNING_UPDATED handler is never registered, so the
# projection only ever ran from a manual POST /intelligence/project — which is why
# intelligence_insights sat at 0 rows and Adaptive had nothing to propose from.
from navis.intelligence import subscribers as _intelligence_subscribers  # noqa: F401  (LEARNING_UPDATED)
from navis.execution import worker as _execution_worker  # noqa: F401  (registers tasks)
from navis.signals import outbox as _signal_outbox  # noqa: F401  (registers the outbox relay task)
from navis.signals import importance_backfill as _signal_importance_backfill  # noqa: F401  (scoring backfill task)
from navis.signals import token_backfill as _signal_token_backfill  # noqa: F401  (one-shot token re-encryption task)
from navis.memory import outbox as _memory_outbox  # noqa: F401  (registers the memory outbox relay task)
from navis.identity import maintenance as _identity_maintenance  # noqa: F401  (registers purge tasks)
from navis.observation import subscribers as _observation_subscribers  # noqa: F401  (closes the loop)
from navis.orchestration import org_provisioning as _org_provisioning  # noqa: F401  (ORG_CREATED + onboarding funnel)
# The beat schedule names these tasks, but a task only exists on the worker if its
# module was imported. Without these two lines the worker rejected every dispatch
# with "Received unregistered task of type navis.orchestration.sync.sync_all_sources",
# so Gmail was never polled (no new signal_events), Gmail watches were never renewed
# (push delivery expires after ~7 days), and the morning brief never rebuilt.
# tests/platform/test_outbox_delivery.py asserts every beat-scheduled task resolves.
from navis.orchestration import sync as _orchestration_sync  # noqa: F401  (sync_all_sources, renew_watches)
from navis.orchestration import brief as _orchestration_brief  # noqa: F401  (build_all_briefs)
from navis.ml.embeddings import subscribers as _ml_embedding_subscribers  # noqa: F401  (real-time embed of new memory)
from navis.ml.embeddings import jobs as _ml_embedding_jobs  # noqa: F401  (embed_org backfill task)
