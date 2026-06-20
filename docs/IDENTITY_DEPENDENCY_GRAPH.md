# Identity Dependency Graph (Phase 1)

```
            Platform  (config, database, security, events, monitoring)
                │  (Identity imports ONLY this + itself)
                ▼
            Identity   ← the foundation
                │  (imported downward by everyone above it)
                ▼
   Organization → Signals → Memory → State → Reasoning → Decision
   → Execution → Observation → Learning → Intelligence → Adaptive
   (+ ML, Orchestration, API as cross-cutting consumers)
```

## Verification (grounded in code)

**Identity imports ONLY platform + identity.** Verified by AST/grep over `navis/identity/*.py`:

```
from navis.platform.*   ← allowed (foundation infra)
from navis.identity.*   ← internal
```

Grep for any forbidden import
(`signals|memory|state|reasoning|decisions|execution|observation|learning|intelligence|adaptive|organization|ml|orchestration`)
in `navis/identity/` returns **zero matches**. There is also a CI guard test
(`tests/identity/test_identity_v2.py::test_identity_imports_no_business_domain`)
that fails the build on any violation.

| Check | Result |
|---|---|
| Upward dependency (identity → business domain) | **None** |
| Hidden coupling | **None** (only `platform.events` for the bus, `platform.database/security/config`) |
| Import inversion | **None** |
| Circular dependency | **None** (organization → identity is the only cross-edge; never the reverse) |

**Conclusion:** the dependency law holds perfectly. Identity is a clean foundation layer. No Phase-1 violations to fix.
