"""State — Layer 3. The only read model between Memory and Reasoning.

The production implementation is `navis.state.v2`: a deterministic, disposable,
versioned projection over `memory_records`. `OrganizationalState` is rebuilt from
Memory at any time (snapshots are a cache, never a source of truth). Import from
`navis.state.v2`:

    from navis.state.v2 import StateService, OrganizationalState
"""
