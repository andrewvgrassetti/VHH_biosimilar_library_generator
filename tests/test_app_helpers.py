"""Smoke tests for app-layer helpers introduced by the design-system update.

These tests exercise the pure-logic helpers without launching Streamlit.
They validate:
- RuntimeConfig construction from sidebar-like dicts
- DesignPolicy round-trip serialisation (JSON)
- PositionClassifier → DesignPolicy integration
- Policy import/export contract
- Three-state interactive selector ↔ policy synchronisation
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
from vhh_library.utils import AMINO_ACIDS, DEFAULT_CONSERVATIVE_FALLBACK, SIMILAR_AA_GROUPS

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
    """Reproduce the *legacy* binary-off-limits policy-building logic.

    This helper matches the original app.py behaviour prior to the three-state
    selector.  Existing tests that depend on it are kept for regression safety.
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


def _build_policy_from_three_state(
    imgt_keys: list[str],
    frozen_positions: set[str],
    conservative_positions: set[str],
    imgt_numbered: dict[str, str] | None = None,
    position_forbidden: dict[str, set[str]] | None = None,
) -> DesignPolicy:
    """Reproduce the three-state policy-building logic from app.py.

    This mirrors the exact logic in the updated ``tab_mutations()``:
    1. Use classifier for allowed-AA metadata.
    2. Build policy directly from frozen/conservative/mutable sets.
    3. Apply CSV forbidden substitutions **last** (ultimate arbiter).

    Parameters
    ----------
    imgt_keys : list[str]
        All IMGT position keys in the sequence.
    frozen_positions : set[str]
        Positions the user marked as frozen.
    conservative_positions : set[str]
        Positions the user marked as conservative.
    imgt_numbered : dict[str, str], optional
        IMGT position → wild-type AA mapping.  If ``None``, uses ``"A"``
        as the fallback WT residue for similarity lookups.
    position_forbidden : dict[str, set[str]], optional
        CSV-derived forbidden substitutions (position → set of forbidden AAs).
        Applied last and overrides all other settings.

    Returns
    -------
    DesignPolicy
    """
    classifier = PositionClassifier()
    classifications = classifier.classify(imgt_keys)

    policy = DesignPolicy()
    for pos_key in imgt_keys:
        if pos_key in frozen_positions:
            policy.freeze([pos_key])
        elif pos_key in conservative_positions:
            clf = classifications.get(pos_key)
            if clf and clf.position_class is PositionClass.CONSERVATIVE and clf.allowed_aas:
                policy.restrict(pos_key, clf.allowed_aas)
            else:
                wt_aa = (imgt_numbered or {}).get(pos_key, "A")
                similar = SIMILAR_AA_GROUPS.get(wt_aa, DEFAULT_CONSERVATIVE_FALLBACK)
                policy.restrict(pos_key, similar)
        else:
            policy.make_mutable([pos_key])

    # CSV forbidden substitutions override everything — applied last.
    if position_forbidden:
        for pos_key, forbidden_set in position_forbidden.items():
            allowed = AMINO_ACIDS - frozenset(forbidden_set)
            if allowed:
                policy.restrict(pos_key, allowed)
            else:
                policy.freeze([pos_key])

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


# ---------------------------------------------------------------------------
# Three-state selector → policy (new system)
# ---------------------------------------------------------------------------


class TestThreeStatePolicyFromSelector:
    """Tests for the three-state interactive selector → policy pipeline.

    Validates that the new three-state system (frozen/conservative/mutable)
    correctly builds a DesignPolicy matching the selector's authoritative state,
    and that the CSV forbidden-substitutions override everything.
    """

    @pytest.fixture()
    def imgt_keys(self) -> list[str]:
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
            "104",  # conserved Cys
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
    def default_frozen(self, imgt_keys: list[str]) -> set[str]:
        """Default frozen set from classifier (CDRs + conserved)."""
        classifier = PositionClassifier()
        classifications = classifier.classify(imgt_keys)
        return {k for k, v in classifications.items() if v.position_class is PositionClass.FROZEN}

    @pytest.fixture()
    def default_conservative(self, imgt_keys: list[str]) -> set[str]:
        """Default conservative set from classifier."""
        classifier = PositionClassifier()
        classifications = classifier.classify(imgt_keys)
        return {k for k, v in classifications.items() if v.position_class is PositionClass.CONSERVATIVE}

    def test_default_classes_match_classifier(
        self, imgt_keys: list[str], default_frozen: set[str], default_conservative: set[str]
    ):
        """When classes match classifier defaults, policy should be identical."""
        policy = _build_policy_from_three_state(imgt_keys, default_frozen, default_conservative)
        for pos in default_frozen:
            assert policy.effective_class(pos) is PositionClass.FROZEN
        for pos in default_conservative:
            assert policy.effective_class(pos) is PositionClass.CONSERVATIVE

    def test_user_toggles_mutable_to_frozen(
        self, imgt_keys: list[str], default_frozen: set[str], default_conservative: set[str]
    ):
        """User clicking a mutable position should freeze it."""
        expanded_frozen = default_frozen | {"10"}
        policy = _build_policy_from_three_state(imgt_keys, expanded_frozen, default_conservative)
        assert policy.effective_class("10") is PositionClass.FROZEN

    def test_user_toggles_frozen_to_conservative(
        self, imgt_keys: list[str], default_frozen: set[str], default_conservative: set[str]
    ):
        """User clicking a frozen CDR position should make it conservative."""
        reduced_frozen = default_frozen - {"30"}
        expanded_conservative = default_conservative | {"30"}
        policy = _build_policy_from_three_state(imgt_keys, reduced_frozen, expanded_conservative)
        assert policy.effective_class("30") is PositionClass.CONSERVATIVE

    def test_user_toggles_conservative_to_mutable(
        self, imgt_keys: list[str], default_frozen: set[str], default_conservative: set[str]
    ):
        """User clicking a conservative position should make it mutable."""
        reduced_conservative = default_conservative - {"42"}
        policy = _build_policy_from_three_state(imgt_keys, default_frozen, reduced_conservative)
        assert policy.effective_class("42") is PositionClass.MUTABLE

    def test_conserved_positions_stay_frozen_when_included(
        self, imgt_keys: list[str], default_frozen: set[str], default_conservative: set[str]
    ):
        """Conserved Cys/Trp remain frozen when in the frozen set."""
        policy = _build_policy_from_three_state(imgt_keys, default_frozen, default_conservative)
        assert policy.effective_class("23") is PositionClass.FROZEN
        assert policy.effective_class("41") is PositionClass.FROZEN
        assert policy.effective_class("104") is PositionClass.FROZEN

    def test_conservative_uses_classifier_allowed_aas(
        self, imgt_keys: list[str], default_frozen: set[str], default_conservative: set[str]
    ):
        """Conservative positions with classifier data should use those AAs."""
        policy = _build_policy_from_three_state(imgt_keys, default_frozen, default_conservative)
        pp = policy["42"]
        assert pp.is_conservative
        assert pp.allowed_aas is not None
        assert "F" in pp.allowed_aas or "Y" in pp.allowed_aas

    def test_conservative_uses_similar_aas_for_non_classifier_positions(
        self, imgt_keys: list[str], default_frozen: set[str], default_conservative: set[str]
    ):
        """Conservative positions without classifier data should use similarity groups."""
        expanded_conservative = default_conservative | {"10"}
        imgt_numbered = {k: "A" for k in imgt_keys}
        policy = _build_policy_from_three_state(imgt_keys, default_frozen, expanded_conservative, imgt_numbered)
        pp = policy["10"]
        assert pp.is_conservative
        assert pp.allowed_aas == SIMILAR_AA_GROUPS["A"]

    def test_full_cycle_mutable_frozen_conservative_mutable(
        self, imgt_keys: list[str], default_frozen: set[str], default_conservative: set[str]
    ):
        """Simulate full click cycle: mutable→frozen→conservative→mutable."""
        # Start: position "1" is mutable
        policy = _build_policy_from_three_state(imgt_keys, default_frozen, default_conservative)
        assert policy.effective_class("1") is PositionClass.MUTABLE

        # Click 1: mutable → frozen
        frozen1 = default_frozen | {"1"}
        policy = _build_policy_from_three_state(imgt_keys, frozen1, default_conservative)
        assert policy.effective_class("1") is PositionClass.FROZEN

        # Click 2: frozen → conservative
        frozen2 = frozen1 - {"1"}
        conservative2 = default_conservative | {"1"}
        policy = _build_policy_from_three_state(imgt_keys, frozen2, conservative2)
        assert policy.effective_class("1") is PositionClass.CONSERVATIVE

        # Click 3: conservative → mutable
        conservative3 = conservative2 - {"1"}
        policy = _build_policy_from_three_state(imgt_keys, frozen2, conservative3)
        assert policy.effective_class("1") is PositionClass.MUTABLE

    def test_empty_sets_means_all_mutable(self, imgt_keys: list[str]):
        """With both sets empty, all positions should be mutable."""
        policy = _build_policy_from_three_state(imgt_keys, set(), set())
        for pos in imgt_keys:
            assert policy.effective_class(pos) is PositionClass.MUTABLE


# ---------------------------------------------------------------------------
# CSV forbidden substitutions override all other settings
# ---------------------------------------------------------------------------


class TestCSVForbiddenSubstitutionsOverride:
    """Verify that the CSV forbidden-substitutions file is the ultimate arbiter.

    The CSV restrictions must override interactive selector state, classifier
    defaults, and any other settings.  They are applied last in the pipeline.
    """

    @pytest.fixture()
    def imgt_keys(self) -> list[str]:
        return [
            "1",
            "2",
            "3",
            "10",
            "20",
            "23",  # conserved Cys
            "42",  # FR2 hallmark (conservative by default)
            "56",  # CDR2
            "69",
            "78",
            "104",  # conserved Cys
            "118",  # FR4
        ]

    @pytest.fixture()
    def imgt_numbered(self, imgt_keys: list[str]) -> dict[str, str]:
        """Fake IMGT numbered dict."""
        aa_map = {
            "1": "E",
            "2": "V",
            "3": "Q",
            "10": "G",
            "20": "L",
            "23": "C",
            "42": "F",
            "56": "I",
            "69": "K",
            "78": "A",
            "104": "C",
            "118": "W",
        }
        return {k: aa_map.get(k, "A") for k in imgt_keys}

    def test_csv_overrides_mutable_to_conservative(self, imgt_keys: list[str]):
        """CSV makes a mutable position conservative (restricts allowed AAs)."""
        # Position "10" is mutable by default
        policy = _build_policy_from_three_state(
            imgt_keys,
            set(),
            set(),
            position_forbidden={"10": {"P", "C", "W"}},
        )
        assert policy.effective_class("10") is PositionClass.CONSERVATIVE
        pp = policy["10"]
        assert "P" not in pp.allowed_aas
        assert "C" not in pp.allowed_aas
        assert "W" not in pp.allowed_aas
        # Other AAs should still be allowed
        assert "A" in pp.allowed_aas

    def test_csv_overrides_selector_frozen_to_conservative(self, imgt_keys: list[str]):
        """CSV can restrict a frozen position to conservative (allows some AAs)."""
        # User froze position "10", but CSV says only some AAs are forbidden
        policy = _build_policy_from_three_state(
            imgt_keys,
            {"10"},
            set(),
            position_forbidden={"10": {"P", "C"}},
        )
        # CSV overrides: position becomes conservative (restricted), not frozen
        assert policy.effective_class("10") is PositionClass.CONSERVATIVE
        pp = policy["10"]
        assert "P" not in pp.allowed_aas
        assert "C" not in pp.allowed_aas

    def test_csv_freezes_when_all_aas_forbidden(self, imgt_keys: list[str]):
        """When CSV forbids all 20 AAs, position becomes frozen."""
        policy = _build_policy_from_three_state(
            imgt_keys,
            set(),
            set(),
            position_forbidden={"10": set(AMINO_ACIDS)},
        )
        assert policy.effective_class("10") is PositionClass.FROZEN

    def test_csv_overrides_selector_conservative(self, imgt_keys: list[str], imgt_numbered: dict[str, str]):
        """CSV restrictions are additive on top of conservative AA sets."""
        # Make "10" conservative via selector
        policy_no_csv = _build_policy_from_three_state(
            imgt_keys,
            set(),
            {"10"},
            imgt_numbered,
        )
        similar_for_g = SIMILAR_AA_GROUPS["G"]  # G is WT at position 10
        assert policy_no_csv["10"].allowed_aas == similar_for_g

        # Now add CSV that further restricts
        policy_with_csv = _build_policy_from_three_state(
            imgt_keys,
            set(),
            {"10"},
            imgt_numbered,
            position_forbidden={"10": {"A", "S"}},
        )
        pp = policy_with_csv["10"]
        assert pp.is_conservative
        # CSV-forbidden AAs must not be in allowed set
        assert "A" not in pp.allowed_aas
        assert "S" not in pp.allowed_aas

    def test_csv_does_not_affect_unmentioned_positions(self, imgt_keys: list[str]):
        """Positions not in the CSV retain their selector/classifier class."""
        policy = _build_policy_from_three_state(
            imgt_keys,
            {"23"},
            {"42"},
            position_forbidden={"10": {"P"}},
        )
        # Unmentioned positions: class unchanged
        assert policy.effective_class("23") is PositionClass.FROZEN
        assert policy.effective_class("42") is PositionClass.CONSERVATIVE
        assert policy.effective_class("1") is PositionClass.MUTABLE
        # CSV-affected position
        assert policy.effective_class("10") is PositionClass.CONSERVATIVE

    def test_csv_applied_last_overrides_all_layers(self, imgt_keys: list[str]):
        """CSV is applied after selector and classifier, overriding both."""
        # Position "42" is conservative by classifier.  User left it conservative.
        # CSV forbids "F" and "Y" at position 42.
        policy = _build_policy_from_three_state(
            imgt_keys,
            set(),
            {"42"},
            position_forbidden={"42": {"F", "Y"}},
        )
        pp = policy["42"]
        assert pp.is_conservative
        assert "F" not in pp.allowed_aas
        assert "Y" not in pp.allowed_aas
