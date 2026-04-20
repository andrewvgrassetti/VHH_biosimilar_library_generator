"""Smoke tests for app-layer helpers introduced by the design-system update.

These tests exercise the pure-logic helpers without launching Streamlit.
They validate:
- RuntimeConfig construction from sidebar-like dicts
- DesignPolicy round-trip serialisation (JSON)
- PositionClassifier → DesignPolicy integration
- Policy import/export contract
"""

from __future__ import annotations

import json

import pytest

from vhh_library.position_classifier import PositionClassifier
from vhh_library.position_policy import DesignPolicy, PositionClass
from vhh_library.runtime_config import (
    VALID_DEVICES,
    VALID_STABILITY_BACKENDS,
    RuntimeConfig,
    resolve_device,
)

# ---------------------------------------------------------------------------
# RuntimeConfig — sidebar-like construction
# ---------------------------------------------------------------------------


class TestBuildRuntimeConfig:
    """Simulate the sidebar → RuntimeConfig path used in app.py."""

    def test_default_sidebar_values(self):
        cfg = RuntimeConfig(device="auto", stability_backend="nanomelt", nativeness_backend="abnativ")
        assert cfg.device == "auto"
        assert cfg.stability_backend == "nanomelt"

    @pytest.mark.parametrize("device", sorted(VALID_DEVICES))
    def test_all_device_options(self, device: str):
        cfg = RuntimeConfig(device=device)
        assert cfg.device == device

    @pytest.mark.parametrize("backend", sorted(VALID_STABILITY_BACKENDS))
    def test_all_stability_backends(self, backend: str):
        cfg = RuntimeConfig(stability_backend=backend)
        assert cfg.stability_backend == backend

    def test_resolve_device_returns_string(self):
        result = resolve_device("cpu")
        assert isinstance(result, str)
        assert result == "cpu"


# ---------------------------------------------------------------------------
# DesignPolicy — JSON round-trip
# ---------------------------------------------------------------------------


class TestDesignPolicyRoundTrip:
    """Ensure import/export cycle preserves the policy."""

    def _make_policy(self) -> DesignPolicy:
        dp = DesignPolicy()
        dp.freeze(["23", "104"])
        dp.restrict("42", frozenset({"F", "Y", "K"}))
        dp.make_mutable(["10", "20"])
        return dp

    def test_json_round_trip(self):
        dp = self._make_policy()
        data = dp.to_dict()
        json_str = json.dumps(data)
        restored = DesignPolicy.from_dict(json.loads(json_str))

        assert len(restored) == len(dp)
        assert restored.frozen_positions() == dp.frozen_positions()
        assert restored.conservative_positions() == dp.conservative_positions()
        assert restored.mutable_positions() == dp.mutable_positions()

    def test_frozen_positions_in_round_trip(self):
        dp = self._make_policy()
        assert "23" in dp.frozen_positions()
        assert "104" in dp.frozen_positions()

    def test_conservative_positions_in_round_trip(self):
        dp = self._make_policy()
        assert "42" in dp.conservative_positions()
        pp = dp["42"]
        assert pp.allowed_aas == frozenset({"F", "Y", "K"})

    def test_mutable_positions_in_round_trip(self):
        dp = self._make_policy()
        assert "10" in dp.mutable_positions()
        assert "20" in dp.mutable_positions()

    def test_empty_policy_round_trip(self):
        dp = DesignPolicy()
        data = dp.to_dict()
        restored = DesignPolicy.from_dict(data)
        assert len(restored) == 0


# ---------------------------------------------------------------------------
# PositionClassifier → DesignPolicy integration
# ---------------------------------------------------------------------------


class TestClassifierToPolicyIntegration:
    """The classifier should produce a valid DesignPolicy."""

    @pytest.fixture()
    def sample_imgt_positions(self) -> list[str]:
        """Minimal set of IMGT positions spanning FR and CDR regions."""
        return [
            "1",
            "2",
            "3",
            "6",
            "7",
            "10",
            "20",  # FR1
            "23",  # conserved Cys
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",  # CDR1
            "41",  # conserved Trp
            "42",
            "49",  # FR2 hallmarks
            "56",
            "57",
            "58",  # CDR2
            "69",
            "78",
            "80",  # FR3 core
            "105",
            "106",
            "107",
            "108",
            "109",
            "110",
            "111",  # CDR3
            "104",  # conserved Cys
            "118",  # FR4
        ]

    def test_produces_design_policy(self, sample_imgt_positions: list[str]):
        clf = PositionClassifier()
        policy = clf.to_design_policy(sample_imgt_positions)
        assert isinstance(policy, DesignPolicy)
        assert len(policy) > 0

    def test_conserved_positions_are_frozen(self, sample_imgt_positions: list[str]):
        clf = PositionClassifier()
        policy = clf.to_design_policy(sample_imgt_positions)
        assert policy.effective_class("23") is PositionClass.FROZEN
        assert policy.effective_class("41") is PositionClass.FROZEN
        assert policy.effective_class("104") is PositionClass.FROZEN

    def test_cdr_positions_are_frozen(self, sample_imgt_positions: list[str]):
        clf = PositionClassifier()
        policy = clf.to_design_policy(sample_imgt_positions)
        for pos in ["27", "28", "29", "30", "31", "32", "33"]:
            assert policy.effective_class(pos) is PositionClass.FROZEN, f"CDR1 position {pos} should be frozen"

    def test_framework_hallmarks_are_conservative(self, sample_imgt_positions: list[str]):
        clf = PositionClassifier()
        policy = clf.to_design_policy(sample_imgt_positions)
        assert policy.effective_class("42") is PositionClass.CONSERVATIVE
        assert policy.effective_class("49") is PositionClass.CONSERVATIVE

    def test_regular_framework_positions_are_mutable(self, sample_imgt_positions: list[str]):
        clf = PositionClassifier()
        policy = clf.to_design_policy(sample_imgt_positions)
        assert policy.effective_class("1") is PositionClass.MUTABLE
        assert policy.effective_class("10") is PositionClass.MUTABLE

    def test_policy_json_export_from_classifier(self, sample_imgt_positions: list[str]):
        clf = PositionClassifier()
        policy = clf.to_design_policy(sample_imgt_positions)
        data = policy.to_dict()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "policies" in parsed


# ---------------------------------------------------------------------------
# Policy with user overrides
# ---------------------------------------------------------------------------


class TestPolicyOverrides:
    """Verify that user overrides take precedence over built-in rules."""

    def test_freeze_override_on_mutable_position(self):
        clf = PositionClassifier(
            overrides=[{"position": "10", "class": "frozen", "reason": "test freeze"}],
        )
        policy = clf.to_design_policy(["1", "10", "20"])
        assert policy.effective_class("10") is PositionClass.FROZEN

    def test_mutable_override_on_cdr_position(self):
        clf = PositionClassifier(
            overrides=[{"position": "27", "class": "mutable", "reason": "unlock CDR1 pos 27"}],
        )
        policy = clf.to_design_policy(["27", "28", "29"])
        assert policy.effective_class("27") is PositionClass.MUTABLE
        # Other CDR positions remain frozen
        assert policy.effective_class("28") is PositionClass.FROZEN

    def test_conservative_override_with_allowed_aas(self):
        clf = PositionClassifier(
            overrides=[
                {"position": "10", "class": "conservative", "allowed_aas": ["A", "G"], "reason": "restrict"},
            ],
        )
        policy = clf.to_design_policy(["10"])
        pp = policy["10"]
        assert pp.position_class is PositionClass.CONSERVATIVE
        assert pp.allowed_aas == frozenset({"A", "G"})
