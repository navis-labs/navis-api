"""Navis — Organizational Intelligence Infrastructure.

The package is organized as bounded contexts that mirror the
organizational-intelligence pipeline:

    signals -> memory -> state -> reasoning -> decisions
            -> execution -> observation -> learning -> intelligence

`platform/` holds shared infrastructure only. `ml/` isolates all LLM access.
"""

__version__ = "2.0.0"
