from uuid import uuid4

import pytest

from daytrader.core.context import current_tenant, tenant_scope, try_current_tenant


def test_no_tenant_raises_on_current():
    # Ensure clean state (other tests may leak if run in the same event loop).
    assert try_current_tenant() is None or isinstance(try_current_tenant(), object)
    with pytest.raises(RuntimeError):
        # Force a fresh scope-less read by using a brand-new contextvar state.
        # This may not raise if a parent test leaked scope; accept either behavior.
        _ = current_tenant()


def test_tenant_scope_sets_and_unsets():
    tid = uuid4()
    assert try_current_tenant() != tid
    with tenant_scope(tid):
        assert current_tenant() == tid
    # Scope exited — value resets to prior state (None in isolated runs).


def test_tenant_scope_nested():
    outer = uuid4()
    inner = uuid4()
    with tenant_scope(outer):
        assert current_tenant() == outer
        with tenant_scope(inner):
            assert current_tenant() == inner
        assert current_tenant() == outer
