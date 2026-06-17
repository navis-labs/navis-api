"""State — Layer 3. The most important moat.

`OrganizationalState` is a living model of the company computed from memory.
Without this layer the decision engine becomes a god object that recomputes the
world on every request. With it, reasoning reads a single coherent snapshot.

    build_organizational_state(org_id) -> OrganizationalState
"""
