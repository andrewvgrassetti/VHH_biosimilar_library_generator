"""Tests for vhh_library.position_classifier — rule-based IMGT position classifier."""

from __future__ import annotations

import json
import textwrap

import pytest

from vhh_library.position_classifier import (
    _CONSERVATIVE_POSITIONS,
    _CONSERVED_POSITIONS,
    ClassificationReason,
    PositionClassification,
    PositionClassifier,
    _classifications_to_policy,
    load_overrides,
)
from vhh_library.position_policy import PositionClass
from vhh_library.utils import AMINO_ACIDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal set of IMGT positions covering all regions.
_ALL_POSITIONS = [str(i) for i in range(1, 129)]

# A subset of CDR positions for quick checks.
_CDR1_POSITIONS = [str(i) for i in range(27, 39)]
_CDR2_POSITIONS = [str(i) for i in range(56, 66)]
_CDR3_POSITIONS = [str(i) for i in range(105, 118)]

# Framework-only positions (not conserved, not conservative).
_PLAIN_FR_POSITIONS = ["1", "2", "3", "4", "5", "10", "15", "20"]


# ===========================================================================
# ClassificationReason
# ===========================================================================


class TestClassificationReason:
    def test_creation(self):
        r = ClassificationReason(rule="cdr_freeze", description="Frozen CDR position")
        assert r.rule == "cdr_freeze"
        assert r.description == "Frozen CDR position"
        assert r.source == "builtin"

    def test_custom_source(self):
        r = ClassificationReason(rule="user_override", description="User says so", source="/tmp/override.yaml")
        assert r.source == "/tmp/override.yaml"

    def test_immutable(self):
        r = ClassificationReason(rule="test", description="test")
        with pytest.raises(AttributeError):
            r.rule = "other"  # type: ignore[misc]


# ===========================================================================
# PositionClassification
# ===========================================================================


class TestPositionClassification:
    def test_frozen_classification(self):
        clf = PositionClassification(
            imgt_position="23",
            position_class=PositionClass.FROZEN,
            reason=ClassificationReason("conserved_residue", "Conserved Cys"),
        )
        assert clf.position_class is PositionClass.FROZEN
        assert clf.allowed_aas is None

    def test_conservative_classification(self):
        clf = PositionClassification(
            imgt_position="42",
            position_class=PositionClass.CONSERVATIVE,
            allowed_aas=frozenset({"F", "Y"}),
            reason=ClassificationReason("framework_support", "Hallmark position"),
        )
        assert clf.position_class is PositionClass.CONSERVATIVE
        assert clf.allowed_aas == frozenset({"F", "Y"})

    def test_mutable_classification(self):
        clf = PositionClassification(
            imgt_position="10",
            position_class=PositionClass.MUTABLE,
            reason=ClassificationReason("default_mutable", "Framework mutable"),
        )
        assert clf.position_class is PositionClass.MUTABLE


# ===========================================================================
# CDR freeze behaviour
# ===========================================================================


class TestCDRFreeze:
    """CDR positions must be FROZEN by default."""

    def test_all_cdr1_frozen(self):
        classifier = PositionClassifier()
        result = classifier.classify(_ALL_POSITIONS)
        for pos in _CDR1_POSITIONS:
            assert result[pos].position_class is PositionClass.FROZEN, f"CDR1 position {pos} should be frozen"
            assert result[pos].reason.rule == "cdr_freeze"

    def test_all_cdr2_frozen(self):
        classifier = PositionClassifier()
        result = classifier.classify(_ALL_POSITIONS)
        for pos in _CDR2_POSITIONS:
            assert result[pos].position_class is PositionClass.FROZEN, f"CDR2 position {pos} should be frozen"

    def test_all_cdr3_frozen(self):
        classifier = PositionClassifier()
        result = classifier.classify(_ALL_POSITIONS)
        for pos in _CDR3_POSITIONS:
            assert result[pos].position_class is PositionClass.FROZEN, f"CDR3 position {pos} should be frozen"

    def test_cdr_freeze_disabled(self):
        classifier = PositionClassifier(freeze_cdrs=False)
        result = classifier.classify(_ALL_POSITIONS)
        # CDR positions should NOT be frozen (unless they are conserved)
        for pos in _CDR1_POSITIONS:
            if pos not in _CONSERVED_POSITIONS:
                assert result[pos].position_class is not PositionClass.FROZEN, (
                    f"CDR1 position {pos} should not be frozen when freeze_cdrs=False"
                )

    def test_cdr_freeze_reason_metadata(self):
        classifier = PositionClassifier()
        result = classifier.classify(["30"])
        assert result["30"].reason.rule == "cdr_freeze"
        assert "CDR" in result["30"].reason.description


# ===========================================================================
# Conserved structural positions
# ===========================================================================


class TestConservedPositions:
    def test_conserved_frozen_by_default(self):
        classifier = PositionClassifier()
        result = classifier.classify(_ALL_POSITIONS)
        for pos in _CONSERVED_POSITIONS:
            assert result[pos].position_class is PositionClass.FROZEN
            assert result[pos].reason.rule == "conserved_residue"

    def test_conserved_freeze_disabled(self):
        classifier = PositionClassifier(freeze_conserved=False, freeze_cdrs=False)
        result = classifier.classify(_ALL_POSITIONS)
        # Position 23 is FR1 → should be mutable when conserved freeze is off
        assert result["23"].position_class is not PositionClass.FROZEN

    def test_conserved_takes_precedence_over_conservative(self):
        """Even if pos 23 were in conservative_positions, conserved wins."""
        classifier = PositionClassifier(
            conservative_positions={"23": frozenset({"C", "S"})},
        )
        result = classifier.classify(["23"])
        assert result["23"].position_class is PositionClass.FROZEN
        assert result["23"].reason.rule == "conserved_residue"


# ===========================================================================
# Conservative allowed-residue handling
# ===========================================================================


class TestConservativePositions:
    def test_builtin_conservative_positions(self):
        classifier = PositionClassifier()
        result = classifier.classify(_ALL_POSITIONS)
        for pos, expected_aas in _CONSERVATIVE_POSITIONS.items():
            assert result[pos].position_class is PositionClass.CONSERVATIVE, f"Position {pos} should be conservative"
            assert result[pos].allowed_aas == expected_aas
            assert result[pos].reason.rule == "framework_support"

    def test_custom_conservative_positions(self):
        custom = {"45": frozenset({"A", "G", "S"})}
        classifier = PositionClassifier(conservative_positions=custom)
        result = classifier.classify(["45"])
        assert result["45"].position_class is PositionClass.CONSERVATIVE
        assert result["45"].allowed_aas == frozenset({"A", "G", "S"})

    def test_conservative_produces_valid_policy(self):
        """Conservative classifications produce valid PositionPolicy objects."""
        classifier = PositionClassifier()
        result = classifier.classify(_ALL_POSITIONS)
        policy = _classifications_to_policy(result)
        for pos in _CONSERVATIVE_POSITIONS:
            pp = policy[pos]
            assert pp.is_conservative
            assert pp.allowed_aas is not None and len(pp.allowed_aas) > 0

    def test_empty_conservative_positions(self):
        """No conservative positions → all framework positions are mutable."""
        classifier = PositionClassifier(conservative_positions={})
        result = classifier.classify(_PLAIN_FR_POSITIONS)
        for pos in _PLAIN_FR_POSITIONS:
            if pos not in _CONSERVED_POSITIONS:
                assert result[pos].position_class is PositionClass.MUTABLE


# ===========================================================================
# Override precedence
# ===========================================================================


class TestOverridePrecedence:
    def test_override_unfreezes_cdr(self):
        """A user override can unfreeze a CDR position."""
        overrides = [{"position": "30", "class": "mutable", "reason": "Unlock CDR1 pos 30"}]
        classifier = PositionClassifier(overrides=overrides)
        result = classifier.classify(_ALL_POSITIONS)
        assert result["30"].position_class is PositionClass.MUTABLE
        assert result["30"].reason.rule == "user_override"

    def test_override_freezes_framework(self):
        """A user override can freeze a normally-mutable framework position."""
        overrides = [{"position": "10", "class": "frozen"}]
        classifier = PositionClassifier(overrides=overrides)
        result = classifier.classify(["10"])
        assert result["10"].position_class is PositionClass.FROZEN
        assert result["10"].reason.rule == "user_override"

    def test_override_restricts_conservative(self):
        """A user override can tighten a conservative position's allowed AAs."""
        overrides = [{"position": "42", "class": "conservative", "allowed_aas": ["F", "Y"]}]
        classifier = PositionClassifier(overrides=overrides)
        result = classifier.classify(["42"])
        assert result["42"].position_class is PositionClass.CONSERVATIVE
        assert result["42"].allowed_aas == frozenset({"F", "Y"})

    def test_override_unfreezes_conserved(self):
        """A user override can override even conserved positions."""
        overrides = [{"position": "23", "class": "mutable", "reason": "Expert override"}]
        classifier = PositionClassifier(overrides=overrides)
        result = classifier.classify(["23"])
        assert result["23"].position_class is PositionClass.MUTABLE

    def test_later_override_wins(self):
        """When multiple overrides target the same position, last one wins."""
        overrides = [
            {"position": "10", "class": "frozen"},
            {"position": "10", "class": "mutable", "reason": "Changed mind"},
        ]
        classifier = PositionClassifier(overrides=overrides)
        result = classifier.classify(["10"])
        assert result["10"].position_class is PositionClass.MUTABLE
        assert result["10"].reason.description == "Changed mind"

    def test_override_adds_position_not_in_sequence(self):
        """Overrides can classify positions not in the input list."""
        overrides = [{"position": "128", "class": "frozen"}]
        classifier = PositionClassifier(overrides=overrides)
        result = classifier.classify(["1", "2"])
        assert "128" in result
        assert result["128"].position_class is PositionClass.FROZEN


# ===========================================================================
# Override loading from files
# ===========================================================================


class TestOverrideLoading:
    def test_load_json_overrides(self, tmp_path):
        override_data = {
            "overrides": [
                {"position": "30", "class": "mutable", "reason": "Unlock CDR1"},
                {"position": "60", "class": "conservative", "allowed_aas": ["A", "G"]},
            ]
        }
        p = tmp_path / "overrides.json"
        p.write_text(json.dumps(override_data))

        entries = load_overrides(p)
        assert len(entries) == 2
        assert entries[0]["position"] == "30"

    def test_load_json_classifier_integration(self, tmp_path):
        override_data = {
            "overrides": [
                {"position": "30", "class": "mutable"},
            ]
        }
        p = tmp_path / "overrides.json"
        p.write_text(json.dumps(override_data))

        classifier = PositionClassifier(override_file=p)
        result = classifier.classify(_ALL_POSITIONS)
        assert result["30"].position_class is PositionClass.MUTABLE
        assert result["30"].reason.rule == "user_override"

    def test_load_yaml_overrides(self, tmp_path):
        yaml_text = textwrap.dedent("""\
        overrides:
          - position: "42"
            class: conservative
            allowed_aas:
              - F
              - Y
            reason: Restrict hallmark
        """)
        p = tmp_path / "overrides.yaml"
        p.write_text(yaml_text)

        try:
            entries = load_overrides(p)
            assert len(entries) == 1
            assert entries[0]["position"] == "42"
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_overrides("/nonexistent/path.json")

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "overrides.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported override file format"):
            load_overrides(p)

    def test_missing_overrides_key_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"positions": []}))
        with pytest.raises(ValueError, match="must contain a top-level 'overrides' key"):
            load_overrides(p)

    def test_file_overrides_beat_inline(self, tmp_path):
        """File overrides are appended after inline → file wins on conflicts."""
        inline = [{"position": "10", "class": "frozen"}]
        file_data = {"overrides": [{"position": "10", "class": "mutable"}]}
        p = tmp_path / "overrides.json"
        p.write_text(json.dumps(file_data))

        classifier = PositionClassifier(overrides=inline, override_file=p)
        result = classifier.classify(["10"])
        assert result["10"].position_class is PositionClass.MUTABLE

    def test_source_attribution_inline_vs_file(self, tmp_path):
        """Inline overrides have source='inline', file overrides have file path."""
        inline = [{"position": "1", "class": "frozen"}]
        file_data = {"overrides": [{"position": "2", "class": "frozen"}]}
        p = tmp_path / "overrides.json"
        p.write_text(json.dumps(file_data))

        classifier = PositionClassifier(overrides=inline, override_file=p)
        result = classifier.classify(["1", "2"])
        assert result["1"].reason.source == "inline"
        assert result["2"].reason.source == str(p)


# ===========================================================================
# Override validation
# ===========================================================================


class TestOverrideValidation:
    def test_missing_position_key_raises(self):
        overrides = [{"class": "frozen"}]
        classifier = PositionClassifier(overrides=overrides)
        with pytest.raises(ValueError, match="missing 'position' key"):
            classifier.classify(["1"])

    def test_missing_class_key_raises(self):
        overrides = [{"position": "10"}]
        classifier = PositionClassifier(overrides=overrides)
        with pytest.raises(ValueError, match="missing 'class' key"):
            classifier.classify(["10"])

    def test_invalid_class_raises(self):
        overrides = [{"position": "10", "class": "wiggle"}]
        classifier = PositionClassifier(overrides=overrides)
        with pytest.raises(ValueError, match="invalid class"):
            classifier.classify(["10"])

    def test_conservative_without_aas_raises(self):
        overrides = [{"position": "10", "class": "conservative"}]
        classifier = PositionClassifier(overrides=overrides)
        with pytest.raises(ValueError, match="CONSERVATIVE must specify"):
            classifier.classify(["10"])

    def test_frozen_with_aas_raises(self):
        overrides = [{"position": "10", "class": "frozen", "allowed_aas": ["A"]}]
        classifier = PositionClassifier(overrides=overrides)
        with pytest.raises(ValueError, match="FROZEN must not specify"):
            classifier.classify(["10"])

    def test_mutable_with_aas_raises(self):
        overrides = [{"position": "10", "class": "mutable", "allowed_aas": ["A"]}]
        classifier = PositionClassifier(overrides=overrides)
        with pytest.raises(ValueError, match="MUTABLE must not specify"):
            classifier.classify(["10"])

    def test_unknown_amino_acid_raises(self):
        overrides = [{"position": "10", "class": "conservative", "allowed_aas": ["X"]}]
        classifier = PositionClassifier(overrides=overrides)
        with pytest.raises(ValueError, match="unknown amino acid"):
            classifier.classify(["10"])


# ===========================================================================
# DesignPolicy integration
# ===========================================================================


class TestDesignPolicyIntegration:
    def test_to_design_policy_basic(self):
        classifier = PositionClassifier()
        policy = classifier.to_design_policy(_ALL_POSITIONS)
        # CDR positions frozen
        assert policy["30"].is_frozen
        # Conserved positions frozen
        assert policy["23"].is_frozen
        # Conservative positions
        for pos in _CONSERVATIVE_POSITIONS:
            assert policy[pos].is_conservative
        # Plain framework mutable
        assert policy["1"].is_mutable

    def test_to_design_policy_permits(self):
        classifier = PositionClassifier()
        policy = classifier.to_design_policy(_ALL_POSITIONS)
        # Frozen CDR: permits nothing
        assert not policy.permits("30", "A")
        # Mutable FR: permits any AA
        assert policy.permits("1", "A")
        # Conservative: permits only allowed
        pos42_allowed = _CONSERVATIVE_POSITIONS["42"]
        for aa in AMINO_ACIDS:
            if aa in pos42_allowed:
                assert policy.permits("42", aa)
            else:
                assert not policy.permits("42", aa)

    def test_classifications_to_policy_roundtrip(self):
        """Classifications → DesignPolicy preserves all classes."""
        classifier = PositionClassifier()
        classifications = classifier.classify(_ALL_POSITIONS)
        policy = _classifications_to_policy(classifications)
        for pos_key, clf in classifications.items():
            pp = policy[pos_key]
            assert pp.position_class is clf.position_class
            assert pp.allowed_aas == clf.allowed_aas


# ===========================================================================
# classify_vhh convenience
# ===========================================================================


class TestClassifyVHH:
    def test_with_real_sequence(self, sample_vhh_seq):
        classifier = PositionClassifier()
        result = classifier.classify_vhh(sample_vhh_seq)
        # CDR positions frozen
        for pos_key in sample_vhh_seq.cdr_positions:
            if pos_key in result:
                assert result[pos_key].position_class is PositionClass.FROZEN

    def test_with_invalid_object(self):
        classifier = PositionClassifier()
        with pytest.raises(TypeError, match="Expected a VHHSequence-like object"):
            classifier.classify_vhh("not a sequence")


# ===========================================================================
# Reason metadata
# ===========================================================================


class TestReasonMetadata:
    def test_every_classification_has_reason(self):
        classifier = PositionClassifier()
        result = classifier.classify(_ALL_POSITIONS)
        for pos_key, clf in result.items():
            assert clf.reason is not None
            assert clf.reason.rule != ""
            assert clf.reason.description != ""

    def test_reason_rules_are_expected(self):
        """All built-in reason rules are from the known set."""
        known_rules = {"conserved_residue", "cdr_freeze", "framework_support", "default_mutable"}
        classifier = PositionClassifier()
        result = classifier.classify(_ALL_POSITIONS)
        for clf in result.values():
            assert clf.reason.rule in known_rules

    def test_override_reason_rule(self):
        overrides = [{"position": "10", "class": "frozen", "reason": "My reason"}]
        classifier = PositionClassifier(overrides=overrides)
        result = classifier.classify(["10"])
        assert result["10"].reason.rule == "user_override"
        assert result["10"].reason.description == "My reason"

    def test_override_default_reason(self):
        """Overrides without explicit reason get 'User override'."""
        overrides = [{"position": "10", "class": "frozen"}]
        classifier = PositionClassifier(overrides=overrides)
        result = classifier.classify(["10"])
        assert result["10"].reason.description == "User override"
