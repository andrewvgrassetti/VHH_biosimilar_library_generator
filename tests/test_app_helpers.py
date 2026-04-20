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


# ---------------------------------------------------------------------------
# Policy building from interactive selector (reproduces app.py logic)
# ---------------------------------------------------------------------------


def _build_policy_from_selector(
    imgt_keys: list[str],
    off_limit_positions: set[str],
) -> DesignPolicy:
    """Reproduce the policy-building logic from app.py's tab_mutations.

    This is the exact sequence of operations the app performs:
    1. Build policy from classifier
    2. Freeze off-limit positions
    3. Un-freeze CDR positions the user removed from off-limits

    Returns
    -------
    DesignPolicy
        The combined policy reflecting classifier defaults overlaid with
        interactive-selector overrides.
    """
    classifier = PositionClassifier()
    classifications = classifier.classify(imgt_keys)
    policy = classifier.to_design_policy(imgt_keys)

    if off_limit_positions:
        policy.freeze(off_limit_positions)

    user_mutable = set(imgt_keys) - off_limit_positions
    to_unfreeze = [
        pos_key
        for pos_key in user_mutable
        if (pp := policy.policies.get(pos_key)) is not None
        and pp.is_frozen
        and (clf := classifications.get(pos_key)) is not None
        and clf.reason.rule == "cdr_freeze"
    ]
    if to_unfreeze:
        policy.make_mutable(to_unfreeze)

    return policy


class TestPolicyFromInteractiveSelector:
    """Verify that the policy-building logic in app.py correctly maps
    interactive selector outputs to frozen/conservative/mutable positions.

    These tests reproduce the exact logic from ``tab_mutations()`` to
    prevent regressions like the one where conserved Cys/Trp positions
    were incorrectly unfrozen (see PR #26 follow-up).
    """

    @pytest.fixture()
    def imgt_keys(self) -> list[str]:
        """Positions spanning all regions including conserved residues."""
        return [
            "1",
            "2",
            "3",
            "6",
            "7",
            "10",
            "20",  # FR1
            "23",  # conserved Cys (FR1)
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",  # CDR1
            "41",  # conserved Trp (FR2)
            "42",
            "49",  # FR2 hallmarks
            "56",
            "57",
            "58",  # CDR2
            "69",
            "78",
            "80",  # FR3 core
            "104",  # conserved Cys (FR3)
            "105",
            "106",
            "107",
            "108",
            "109",
            "110",
            "111",  # CDR3
            "118",  # FR4
        ]

    @pytest.fixture()
    def cdr_off_limits(self, imgt_keys: list[str]) -> set[str]:
        """Default off-limit set: CDR positions only."""
        from vhh_library.components.sequence_selector import imgt_key_int_part
        from vhh_library.sequence import IMGT_REGIONS

        result: set[str] = set()
        for region_name, (start, end) in IMGT_REGIONS.items():
            if region_name.startswith("CDR"):
                for key in imgt_keys:
                    if start <= imgt_key_int_part(key) <= end:
                        result.add(key)
        return result

    def test_conserved_positions_stay_frozen_with_default_off_limits(
        self, imgt_keys: list[str], cdr_off_limits: set[str]
    ):
        """Conserved Cys-23, Trp-41, Cys-104 must remain frozen even though
        they are not in the CDR-based off-limit set."""
        policy = _build_policy_from_selector(imgt_keys, cdr_off_limits)
        assert policy.effective_class("23") is PositionClass.FROZEN
        assert policy.effective_class("41") is PositionClass.FROZEN
        assert policy.effective_class("104") is PositionClass.FROZEN

    def test_cdr_positions_frozen_when_in_off_limits(self, imgt_keys: list[str], cdr_off_limits: set[str]):
        """CDR positions should be frozen when present in off_limit_positions."""
        policy = _build_policy_from_selector(imgt_keys, cdr_off_limits)
        for pos in ["27", "28", "29", "30", "31", "32", "33"]:
            assert policy.effective_class(pos) is PositionClass.FROZEN, f"CDR1 position {pos} should be frozen"

    def test_cdr_position_unfrozen_when_removed_from_off_limits(self, imgt_keys: list[str], cdr_off_limits: set[str]):
        """Removing a CDR position from off-limits should make it mutable."""
        reduced = cdr_off_limits - {"30"}
        policy = _build_policy_from_selector(imgt_keys, reduced)
        assert policy.effective_class("30") is PositionClass.MUTABLE
        # Other CDR1 positions remain frozen
        assert policy.effective_class("27") is PositionClass.FROZEN
        assert policy.effective_class("33") is PositionClass.FROZEN

    def test_conserved_stay_frozen_after_cdr_unfreeze(self, imgt_keys: list[str], cdr_off_limits: set[str]):
        """Unfreezing a CDR position must not affect conserved positions."""
        reduced = cdr_off_limits - {"30"}
        policy = _build_policy_from_selector(imgt_keys, reduced)
        assert policy.effective_class("23") is PositionClass.FROZEN
        assert policy.effective_class("41") is PositionClass.FROZEN
        assert policy.effective_class("104") is PositionClass.FROZEN

    def test_conservative_positions_preserved(self, imgt_keys: list[str], cdr_off_limits: set[str]):
        """Conservative framework positions should not be altered by off-limits."""
        policy = _build_policy_from_selector(imgt_keys, cdr_off_limits)
        assert policy.effective_class("42") is PositionClass.CONSERVATIVE
        assert policy.effective_class("49") is PositionClass.CONSERVATIVE
        assert policy.effective_class("118") is PositionClass.CONSERVATIVE

    def test_framework_mutable_stays_mutable(self, imgt_keys: list[str], cdr_off_limits: set[str]):
        """Regular framework positions remain mutable."""
        policy = _build_policy_from_selector(imgt_keys, cdr_off_limits)
        assert policy.effective_class("1") is PositionClass.MUTABLE
        assert policy.effective_class("10") is PositionClass.MUTABLE

    def test_framework_position_added_to_off_limits_becomes_frozen(
        self, imgt_keys: list[str], cdr_off_limits: set[str]
    ):
        """User can freeze a framework position via the interactive selector."""
        expanded = cdr_off_limits | {"10"}
        policy = _build_policy_from_selector(imgt_keys, expanded)
        assert policy.effective_class("10") is PositionClass.FROZEN

    def test_empty_off_limits_preserves_classifier_defaults(self, imgt_keys: list[str]):
        """With no off-limits, conserved positions must remain frozen."""
        policy = _build_policy_from_selector(imgt_keys, set())
        assert policy.effective_class("23") is PositionClass.FROZEN
        assert policy.effective_class("41") is PositionClass.FROZEN
        assert policy.effective_class("104") is PositionClass.FROZEN
        # With empty off-limits, CDR positions become mutable because the
        # user has deselected all region-based freezes.
        for pos in ["27", "30", "33"]:
            assert policy.effective_class(pos) is PositionClass.MUTABLE

    def test_all_cdrs_unchecked_unfreezes_cdr_keeps_conserved(self, imgt_keys: list[str]):
        """Unchecking all CDR checkboxes should unfreeze CDR positions
        but keep conserved structural positions frozen."""
        policy = _build_policy_from_selector(imgt_keys, set())
        # CDR positions unfrozen
        assert policy.effective_class("56") is PositionClass.MUTABLE  # CDR2
        assert policy.effective_class("105") is PositionClass.MUTABLE  # CDR3
        # Conserved still frozen
        assert policy.effective_class("23") is PositionClass.FROZEN
        assert policy.effective_class("104") is PositionClass.FROZEN
