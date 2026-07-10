"""Regression tests for navis/context/exceptions.py — the Context layer's typed failure
surface: one catchable root, stable machine codes, structured payload on UnknownContextTask,
replay-safe (deterministic) construction, and a minimal hierarchy with no dead types.
"""
from __future__ import annotations

import navis.context.exceptions as exc
from navis.context.exceptions import (
    ContextError,
    PermissionConfigurationError,
    UnknownContextTask,
)
from navis.context.tasks import ContextTask


# --- Hierarchy / catchability -----------------------------------------------

def test_every_error_is_one_catchable_layer():
    # An orchestrator can catch the whole layer with the root...
    for cls in (UnknownContextTask, PermissionConfigurationError):
        assert issubclass(cls, ContextError)
    # ...and the root descends from plain Exception (so a normal `except Exception` catches
    # it, and it does NOT bypass Exception the way KeyboardInterrupt/SystemExit do).
    assert issubclass(ContextError, Exception)
    assert Exception in ContextError.__mro__


def test_layer_is_catchable_as_root():
    try:
        raise UnknownContextTask(ContextTask.COMMITMENTS)
    except ContextError as e:  # caught by the layer root, not just its own type
        assert isinstance(e, UnknownContextTask)


# --- Stable machine codes (frozen contract) ---------------------------------

def test_codes_are_stable_and_distinct():
    assert ContextError.code == "context.error"
    assert UnknownContextTask.code == "context.unknown_task"
    assert PermissionConfigurationError.code == "context.permission_misconfigured"
    codes = {ContextError.code, UnknownContextTask.code, PermissionConfigurationError.code}
    assert len(codes) == 3  # no collisions

    # code is readable off an instance for classification without isinstance ladders.
    assert UnknownContextTask(ContextTask.COMMITMENTS).code == "context.unknown_task"


# --- UnknownContextTask: structured payload ---------------------------------

def test_unknown_task_carries_structured_task_attribute():
    err = UnknownContextTask(ContextTask.COMMITMENTS)
    assert err.task is ContextTask.COMMITMENTS  # machine-readable, not just in the message
    assert "commitments" in str(err).lower()  # and still human-readable


def test_unknown_task_construction_is_deterministic():
    # Replay safety: same input -> byte-identical error (no clock/uuid/random).
    a = UnknownContextTask(ContextTask.COMMITMENTS)
    b = UnknownContextTask(ContextTask.COMMITMENTS)
    assert str(a) == str(b) and a.task == b.task and a.code == b.code


# --- PermissionConfigurationError: plain-message backward compatibility ------

def test_permission_config_error_accepts_plain_message():
    # Its two raise sites construct it with just a message — must keep working.
    err = PermissionConfigurationError("allow_all must not also constrain types")
    assert isinstance(err, ContextError)
    assert str(err) == "allow_all must not also constrain types"


# --- Chaining ---------------------------------------------------------------

def test_supports_exception_chaining():
    try:
        try:
            raise ValueError("root cause")
        except ValueError as root:
            raise PermissionConfigurationError("bad policy") from root
    except PermissionConfigurationError as e:
        assert isinstance(e.__cause__, ValueError)  # cause preserved for diagnosis


# --- Minimal surface: no dead types -----------------------------------------

def test_no_dead_exception_types():
    # The previously-unused ContextCacheError (which also documented a guarantee the
    # repository contradicts) was removed. The surface is exactly the live errors.
    public = {
        name
        for name, obj in vars(exc).items()
        if isinstance(obj, type) and issubclass(obj, ContextError)
    }
    assert public == {"ContextError", "UnknownContextTask", "PermissionConfigurationError"}
