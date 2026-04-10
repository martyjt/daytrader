"""Tests for AlgorithmParam and manifest param_defaults."""

from daytrader.algorithms.base import AlgorithmManifest, AlgorithmParam


def test_param_construction():
    p = AlgorithmParam("period", "int", 14, min=2, max=200, description="RSI period")
    assert p.name == "period"
    assert p.type == "int"
    assert p.default == 14
    assert p.min == 2
    assert p.max == 200


def test_param_defaults_method():
    manifest = AlgorithmManifest(
        id="test",
        name="Test",
        params=[
            AlgorithmParam("fast", "int", 9),
            AlgorithmParam("slow", "int", 21),
            AlgorithmParam("threshold", "float", 0.5),
        ],
    )
    defaults = manifest.param_defaults()
    assert defaults == {"fast": 9, "slow": 21, "threshold": 0.5}


def test_param_defaults_empty():
    manifest = AlgorithmManifest(id="test", name="Test")
    assert manifest.param_defaults() == {}


def test_param_with_choices():
    p = AlgorithmParam("mode", "str", "fast", choices=["fast", "slow", "medium"])
    assert p.choices == ["fast", "slow", "medium"]
    assert p.default == "fast"


def test_manifest_backward_compat():
    """Manifest with no params uses empty list default."""
    manifest = AlgorithmManifest(id="test", name="Test", version="1.0.0")
    assert manifest.params == []
    assert manifest.param_defaults() == {}
