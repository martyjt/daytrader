"""Per-tenant overlay tests for AlgorithmRegistry."""

from __future__ import annotations

from uuid import UUID, uuid4

from daytrader.algorithms.base import Algorithm, AlgorithmManifest
from daytrader.algorithms.registry import AlgorithmRegistry


class _Stub(Algorithm):
    def __init__(self, algo_id: str, name: str = "stub"):
        self._id = algo_id
        self._name = name

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(id=self._id, name=self._name)

    def on_bar(self, ctx):
        return None


def _reset():
    AlgorithmRegistry.clear()


def setup_function():
    _reset()


def teardown_function():
    _reset()


def test_global_get_unaffected_by_overlay():
    AlgorithmRegistry.register(_Stub("global_algo"))
    tid = uuid4()
    AlgorithmRegistry.register_for_tenant(tid, _Stub("plugin_algo"))
    assert AlgorithmRegistry.get("global_algo").manifest.id == "global_algo"


def test_overlay_takes_precedence_over_global():
    AlgorithmRegistry.register(_Stub("shared", name="builtin"))
    tid = uuid4()
    AlgorithmRegistry.register_for_tenant(tid, _Stub("shared", name="plugin"))
    # With tenant_id, overlay wins.
    assert AlgorithmRegistry.get("shared", tenant_id=tid).manifest.name == "plugin"
    # Without tenant_id, the built-in stays.
    assert AlgorithmRegistry.get("shared").manifest.name == "builtin"


def test_overlay_invisible_to_other_tenants():
    a, b = uuid4(), uuid4()
    AlgorithmRegistry.register_for_tenant(a, _Stub("a_only"))
    # Tenant B can't reach tenant A's plugin.
    try:
        AlgorithmRegistry.get("a_only", tenant_id=b)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    # And without tenant_id at all, it's not visible either.
    try:
        AlgorithmRegistry.get("a_only")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_available_includes_overlay_when_tenant_given():
    AlgorithmRegistry.register(_Stub("global"))
    tid = uuid4()
    AlgorithmRegistry.register_for_tenant(tid, _Stub("plugin"))
    assert "global" in AlgorithmRegistry.available()
    assert "plugin" not in AlgorithmRegistry.available()
    assert "plugin" in AlgorithmRegistry.available(tenant_id=tid)
    assert "global" in AlgorithmRegistry.available(tenant_id=tid)


def test_unregister_for_tenant():
    tid = uuid4()
    AlgorithmRegistry.register_for_tenant(tid, _Stub("p1"))
    assert AlgorithmRegistry.unregister_for_tenant(tid, "p1") is True
    assert AlgorithmRegistry.unregister_for_tenant(tid, "p1") is False  # already gone


def test_clear_tenant_drops_all_plugins():
    tid = uuid4()
    AlgorithmRegistry.register_for_tenant(tid, _Stub("p1"))
    AlgorithmRegistry.register_for_tenant(tid, _Stub("p2"))
    AlgorithmRegistry.clear_tenant(tid)
    assert AlgorithmRegistry.tenant_overlay(tid) == {}


def test_get_lists_tenant_overlay_in_error_message():
    AlgorithmRegistry.register(_Stub("global"))
    tid = uuid4()
    AlgorithmRegistry.register_for_tenant(tid, _Stub("their_plugin"))
    try:
        AlgorithmRegistry.get("missing", tenant_id=tid)
    except KeyError as exc:
        msg = str(exc)
        assert "their_plugin" in msg or "global" in msg


def test_clear_resets_overlays_too():
    AlgorithmRegistry.register(_Stub("global"))
    tid = uuid4()
    AlgorithmRegistry.register_for_tenant(tid, _Stub("plugin"))
    AlgorithmRegistry.clear()
    assert AlgorithmRegistry.available() == []
    assert AlgorithmRegistry.available(tenant_id=tid) == []
