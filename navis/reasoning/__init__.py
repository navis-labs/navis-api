"""Reasoning — Layer 4. Intelligence engines.

Each engine reads the OrganizationalState and emits ReasoningSignals (risks,
opportunities, priorities). Engines are pure: state in, signals out. They never
touch the database or the outside world — that makes them cheap to test and
compose. The decision layer ranks their output.
"""
from navis.reasoning.base import ReasoningSignal, SignalKind
from navis.reasoning.service import run_reasoning

__all__ = ["ReasoningSignal", "SignalKind", "run_reasoning"]
