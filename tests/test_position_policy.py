"""Tests for vhh_library.position_policy — three-class position design model."""

from __future__ import annotations

import pytest

from vhh_library.position_policy import (
    DesignPolicy,
    PositionClass,
    PositionPolicy,
    default_design_policy,
    from_off_limits,
    from_vhh_sequence,
    imgt_base_number,
    imgt_region_for,
    parse_imgt_position,
    to_off_limits,
)
from vhh_library.utils import AMINO_ACIDS

# ===========================================================================
# parse_imgt_position
# ===========================================================================


class TestParseImgtPosition:
    def test_integer_input(self):
        assert parse_imgt_position(1) == "1"
        assert parse_imgt_position(111) == "111"

    def test_string_integer_input(self):
        assert parse_imgt_position("1") == "1"
        assert parse_imgt_position("111") == "111"

    def test_insertion_code(self):
        assert parse_imgt_position("111A") == "111A"
        assert parse_imgt_position("111B") == "111B"

    def test_leading_zeros_stripped(self):
        assert parse_imgt_position("01") == "1"
        assert parse_imgt_position("0111A") == "111A"

    def test_whitespace_stripped(self):
        assert parse_imgt_position("  27  ") == "27"

    def test_negative_integer_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_imgt_position(-1)

    def test_zero_integer_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_imgt_position(0)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid IMGT position"):
            parse_imgt_position("abc")

    def test_double_insertion_code_raises(self):
        with pytest.raises(ValueError, match="Invalid IMGT position"):
            parse_imgt_position("111AB")


# ===========================================================================
# imgt_base_number
# ===========================================================================


class TestImgtBaseNumber:
    def test_plain_number(self):
        assert imgt_base_number("27") == 27

    def test_insertion_position(self):
        assert imgt_base_number("111A") == 111
        assert imgt_base_number("111B") == 111

    def test_single_digit(self):
        assert imgt_base_number("1") == 1


# ===========================================================================
# imgt_region_for
# ===========================================================================


class TestImgtRegionFor:
    def test_framework_positions(self):
        assert imgt_region_for("1") == "FR1"
        assert imgt_region_for("25") == "FR1"
        assert imgt_region_for("36") == "FR2"

    def test_cdr_positions(self):
        assert imgt_region_for("26") == "CDR1"
        assert imgt_region_for("35") == "CDR1"
        assert imgt_region_for("50") == "CDR2"

    def test_insertion_position_uses_base(self):
        # 111 is in FR4 (111–128)
        assert imgt_region_for("111A") == "FR4"
        assert imgt_region_for("111B") == "FR4"

    def test_out_of_range_returns_none(self):
        assert imgt_region_for("200") is None
        assert imgt_region_for("0") is None


# ===========================================================================
# PositionClass enum
# ===========================================================================


class TestPositionClass:
    def test_values(self):
        assert PositionClass.FROZEN.value == "frozen"
        assert PositionClass.CONSERVATIVE.value == "conservative"
        assert PositionClass.MUTABLE.value == "mutable"

    def test_from_value(self):
        assert PositionClass("frozen") is PositionClass.FROZEN
        assert PositionClass("mutable") is PositionClass.MUTABLE

    def test_members_count(self):
        assert len(PositionClass) == 3


# ===========================================================================
# PositionPolicy
# ===========================================================================


class TestPositionPolicy:
    def test_frozen_creation(self):
        pp = PositionPolicy("23", PositionClass.FROZEN)
        assert pp.is_frozen
        assert not pp.is_mutable
        assert not pp.is_conservative
        assert pp.allowed_aas is None

    def test_mutable_creation(self):
        pp = PositionPolicy("45", PositionClass.MUTABLE)
        assert pp.is_mutable
        assert pp.allowed_aas is None

    def test_conservative_creation(self):
        pp = PositionPolicy("60", PositionClass.CONSERVATIVE, frozenset({"A", "G", "S"}))
        assert pp.is_conservative
        assert pp.allowed_aas == {"A", "G", "S"}

    def test_frozen_with_allowed_aas_raises(self):
        with pytest.raises(ValueError, match="FROZEN positions must not specify allowed_aas"):
            PositionPolicy("23", PositionClass.FROZEN, frozenset({"A"}))

    def test_conservative_without_allowed_aas_raises(self):
        with pytest.raises(ValueError, match="CONSERVATIVE positions must specify"):
            PositionPolicy("60", PositionClass.CONSERVATIVE)

    def test_conservative_with_empty_aas_raises(self):
        with pytest.raises(ValueError, match="CONSERVATIVE positions must specify"):
            PositionPolicy("60", PositionClass.CONSERVATIVE, frozenset())

    def test_conservative_with_invalid_aa_raises(self):
        with pytest.raises(ValueError, match="Unknown amino acid"):
            PositionPolicy("60", PositionClass.CONSERVATIVE, frozenset({"X", "A"}))

    def test_mutable_with_allowed_aas_raises(self):
        with pytest.raises(ValueError, match="MUTABLE positions must not specify allowed_aas"):
            PositionPolicy("60", PositionClass.MUTABLE, frozenset({"A"}))

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError, match="Invalid IMGT position"):
            PositionPolicy("abc", PositionClass.FROZEN)

    def test_insertion_position_accepted(self):
        pp = PositionPolicy("111A", PositionClass.FROZEN)
        assert pp.imgt_position == "111A"

    # -- permits ----------------------------------------------------------

    def test_frozen_permits_nothing(self):
        pp = PositionPolicy("23", PositionClass.FROZEN)
        assert not pp.permits("A")
        assert not pp.permits("C")

    def test_conservative_permits_allowed_only(self):
        pp = PositionPolicy("60", PositionClass.CONSERVATIVE, frozenset({"A", "G"}))
        assert pp.permits("A")
        assert pp.permits("G")
        assert not pp.permits("W")

    def test_mutable_permits_any_amino_acid(self):
        pp = PositionPolicy("45", PositionClass.MUTABLE)
        for aa in AMINO_ACIDS:
            assert pp.permits(aa)

    # -- serialisation ----------------------------------------------------

    def test_roundtrip_frozen(self):
        pp = PositionPolicy("23", PositionClass.FROZEN)
        assert PositionPolicy.from_dict(pp.to_dict()) == pp

    def test_roundtrip_conservative(self):
        pp = PositionPolicy("60", PositionClass.CONSERVATIVE, frozenset({"A", "G"}))
        restored = PositionPolicy.from_dict(pp.to_dict())
        assert restored == pp

    def test_roundtrip_mutable(self):
        pp = PositionPolicy("45", PositionClass.MUTABLE)
        assert PositionPolicy.from_dict(pp.to_dict()) == pp

    def test_to_dict_includes_allowed_aas_sorted(self):
        pp = PositionPolicy("60", PositionClass.CONSERVATIVE, frozenset({"G", "A", "S"}))
        d = pp.to_dict()
        assert d["allowed_aas"] == ["A", "G", "S"]

    def test_to_dict_excludes_allowed_aas_for_frozen(self):
        pp = PositionPolicy("23", PositionClass.FROZEN)
        d = pp.to_dict()
        assert "allowed_aas" not in d


# ===========================================================================
# DesignPolicy
# ===========================================================================


class TestDesignPolicy:
    def test_empty_policy(self):
        dp = DesignPolicy()
        assert len(dp) == 0

    def test_set_and_get_policy(self):
        dp = DesignPolicy()
        pp = PositionPolicy("23", PositionClass.FROZEN)
        dp.set_policy(pp)
        assert dp["23"] == pp
        assert "23" in dp

    def test_integer_access(self):
        dp = DesignPolicy()
        dp.set_policy(PositionPolicy("23", PositionClass.FROZEN))
        assert dp[23] == dp["23"]
        assert 23 in dp

    def test_missing_position_returns_none(self):
        dp = DesignPolicy()
        assert dp.get("99") is None

    def test_freeze_multiple(self):
        dp = DesignPolicy()
        dp.freeze([23, "41", 104])
        assert dp["23"].is_frozen
        assert dp["41"].is_frozen
        assert dp["104"].is_frozen

    def test_make_mutable(self):
        dp = DesignPolicy()
        dp.freeze(["27"])
        assert dp["27"].is_frozen
        dp.make_mutable(["27"])
        assert dp["27"].is_mutable

    def test_restrict(self):
        dp = DesignPolicy()
        dp.restrict("60", {"A", "G"})
        assert dp["60"].is_conservative
        assert dp["60"].allowed_aas == frozenset({"A", "G"})

    def test_frozen_positions(self):
        dp = DesignPolicy()
        dp.freeze(["23", "41"])
        dp.set_policy(PositionPolicy("60", PositionClass.MUTABLE))
        assert dp.frozen_positions() == frozenset({"23", "41"})

    def test_conservative_positions(self):
        dp = DesignPolicy()
        dp.restrict("60", {"A"})
        assert dp.conservative_positions() == frozenset({"60"})

    def test_mutable_positions(self):
        dp = DesignPolicy()
        dp.make_mutable(["45", "60"])
        assert dp.mutable_positions() == frozenset({"45", "60"})

    # -- effective_class --------------------------------------------------

    def test_effective_class_explicit(self):
        dp = DesignPolicy()
        dp.freeze(["23"])
        assert dp.effective_class("23") == PositionClass.FROZEN

    def test_effective_class_cdr_default(self):
        dp = DesignPolicy()
        # Position 30 is CDR1
        assert dp.effective_class("30") == PositionClass.FROZEN

    def test_effective_class_framework_default(self):
        dp = DesignPolicy()
        # Position 10 is FR1
        assert dp.effective_class("10") == PositionClass.MUTABLE

    # -- permits ----------------------------------------------------------

    def test_permits_frozen_explicit(self):
        dp = DesignPolicy()
        dp.freeze(["23"])
        assert not dp.permits("23", "A")

    def test_permits_cdr_default(self):
        dp = DesignPolicy()
        assert not dp.permits("30", "A")  # CDR1

    def test_permits_framework_default(self):
        dp = DesignPolicy()
        assert dp.permits("10", "A")  # FR1

    def test_permits_conservative(self):
        dp = DesignPolicy()
        dp.restrict("60", {"A", "G"})
        assert dp.permits("60", "A")
        assert not dp.permits("60", "W")

    # -- insertion positions ----------------------------------------------

    def test_insertion_position_in_policy(self):
        dp = DesignPolicy()
        dp.freeze(["111A"])
        assert "111A" in dp
        assert dp["111A"].is_frozen

    def test_effective_class_insertion_position(self):
        dp = DesignPolicy()
        # 111 is FR4, so default is MUTABLE
        assert dp.effective_class("111A") == PositionClass.MUTABLE

    # -- serialisation ----------------------------------------------------

    def test_roundtrip(self):
        dp = DesignPolicy()
        dp.freeze(["23", "41"])
        dp.restrict("60", {"A", "G"})
        dp.make_mutable(["45"])
        restored = DesignPolicy.from_dict(dp.to_dict())
        assert restored.policies == dp.policies

    # -- repr -------------------------------------------------------------

    def test_repr(self):
        dp = DesignPolicy()
        dp.freeze(["23"])
        dp.restrict("60", {"A"})
        dp.make_mutable(["45"])
        r = repr(dp)
        assert "frozen=1" in r
        assert "conservative=1" in r
        assert "mutable=1" in r


# ===========================================================================
# default_design_policy
# ===========================================================================


class TestDefaultDesignPolicy:
    def test_empty_positions(self):
        dp = default_design_policy([])
        # Only conserved positions should be present
        assert "23" in dp
        assert dp["23"].is_frozen

    def test_cdr_positions_frozen(self):
        positions = [str(i) for i in range(1, 129)]
        dp = default_design_policy(positions)
        # CDR1 positions (26–35) should be frozen
        for pos in range(26, 36):
            assert dp[str(pos)].is_frozen, f"CDR1 position {pos} should be frozen"
        # FR1 positions should be mutable (except conserved)
        assert dp["1"].is_mutable

    def test_conserved_positions_frozen(self):
        positions = [str(i) for i in range(1, 129)]
        dp = default_design_policy(positions)
        assert dp["23"].is_frozen
        assert dp["41"].is_frozen
        assert dp["104"].is_frozen

    def test_freeze_cdrs_false(self):
        positions = [str(i) for i in range(1, 129)]
        dp = default_design_policy(positions, freeze_cdrs=False)
        # CDR positions should be MUTABLE (except conserved ones)
        assert dp["30"].is_mutable

    def test_freeze_conserved_false(self):
        positions = [str(i) for i in range(1, 129)]
        dp = default_design_policy(positions, freeze_conserved=False)
        # Position 23 should be in CDR1 (26–35)? No, 23 is FR1.
        # So with freeze_conserved=False, 23 should be MUTABLE
        assert dp["23"].is_mutable

    def test_insertion_positions_handled(self):
        positions = ["110", "111", "111A", "111B", "112"]
        dp = default_design_policy(positions)
        # 111A and 111B are in FR4 → MUTABLE
        assert dp["111A"].is_mutable
        assert dp["111B"].is_mutable


# ===========================================================================
# Legacy adapters: from_off_limits / to_off_limits
# ===========================================================================


class TestFromOffLimits:
    def test_empty_off_limits(self):
        dp = from_off_limits()
        assert len(dp) == 0

    def test_off_limits_integers(self):
        dp = from_off_limits(off_limits={1, 2, 3})
        assert dp["1"].is_frozen
        assert dp["2"].is_frozen
        assert dp["3"].is_frozen

    def test_off_limits_strings(self):
        dp = from_off_limits(off_limits={"1", "2", "111A"})
        assert dp["1"].is_frozen
        assert dp["111A"].is_frozen

    def test_forbidden_substitutions(self):
        dp = from_off_limits(forbidden_substitutions={"60": {"C", "M"}})
        assert dp["60"].is_conservative
        allowed = dp["60"].allowed_aas
        assert allowed is not None
        assert "C" not in allowed
        assert "M" not in allowed
        assert "A" in allowed

    def test_forbidden_all_aas_becomes_frozen(self):
        dp = from_off_limits(forbidden_substitutions={"60": set(AMINO_ACIDS)})
        assert dp["60"].is_frozen

    def test_off_limits_takes_precedence_over_forbidden(self):
        dp = from_off_limits(
            off_limits={"60"},
            forbidden_substitutions={"60": {"C"}},
        )
        assert dp["60"].is_frozen

    def test_with_imgt_positions_freezes_cdrs(self):
        positions = [str(i) for i in range(1, 129)]
        dp = from_off_limits(imgt_positions=positions, off_limits={"1"})
        assert dp["1"].is_frozen  # explicit
        assert dp["30"].is_frozen  # CDR1 auto-frozen
        assert dp["45"].is_mutable  # FR2

    def test_integer_forbidden_substitution_keys(self):
        dp = from_off_limits(forbidden_substitutions={60: {"C", "M"}})
        assert "60" in dp
        assert dp["60"].is_conservative


class TestToOffLimits:
    def test_frozen_to_off_limits(self):
        dp = DesignPolicy()
        dp.freeze(["23", "41"])
        off, forb = to_off_limits(dp)
        assert off == {"23", "41"}
        assert forb == {}

    def test_conservative_to_forbidden(self):
        dp = DesignPolicy()
        dp.restrict("60", {"A", "G"})
        off, forb = to_off_limits(dp)
        assert "60" not in off
        assert "60" in forb
        assert "A" not in forb["60"]
        assert "G" not in forb["60"]
        # All other AAs should be forbidden
        assert forb["60"] == AMINO_ACIDS - {"A", "G"}

    def test_mutable_produces_nothing(self):
        dp = DesignPolicy()
        dp.make_mutable(["45"])
        off, forb = to_off_limits(dp)
        assert "45" not in off
        assert "45" not in forb


class TestLegacyRoundTrip:
    """Converting legacy → DesignPolicy → legacy should be semantically identical."""

    def test_off_limits_roundtrip(self):
        original_off = {"23", "41", "104"}
        dp = from_off_limits(off_limits=original_off)
        off, forb = to_off_limits(dp)
        assert off == original_off
        assert forb == {}

    def test_forbidden_substitutions_roundtrip(self):
        original_forb = {"60": {"C", "M"}, "70": {"P"}}
        dp = from_off_limits(forbidden_substitutions=original_forb)
        off, forb = to_off_limits(dp)
        assert off == set()
        assert forb == original_forb

    def test_combined_roundtrip(self):
        original_off = {"23"}
        original_forb = {"60": {"C", "M"}}
        dp = from_off_limits(off_limits=original_off, forbidden_substitutions=original_forb)
        off, forb = to_off_limits(dp)
        assert off == original_off
        assert forb == original_forb


# ===========================================================================
# from_vhh_sequence
# ===========================================================================


class TestFromVhhSequence:
    def test_with_real_sequence(self, sample_vhh_seq):
        dp = from_vhh_sequence(sample_vhh_seq)
        # CDR positions should be frozen
        for pos_key in sample_vhh_seq.cdr_positions:
            if pos_key in dp:
                assert dp[pos_key].is_frozen, f"CDR position {pos_key} should be frozen"
        # Conserved positions should be frozen
        assert dp["23"].is_frozen
        assert dp["41"].is_frozen
        assert dp["104"].is_frozen

    def test_with_invalid_object_raises(self):
        with pytest.raises(TypeError, match="Expected a VHHSequence-like object"):
            from_vhh_sequence("not a sequence")

    def test_freeze_cdrs_false(self, sample_vhh_seq):
        dp = from_vhh_sequence(sample_vhh_seq, freeze_cdrs=False)
        # CDR positions should be mutable (unless conserved)
        # Position 30 is CDR1 and not conserved
        if "30" in dp:
            assert dp["30"].is_mutable

    def test_freeze_conserved_false(self, sample_vhh_seq):
        dp = from_vhh_sequence(sample_vhh_seq, freeze_conserved=False, freeze_cdrs=False)
        # Position 23 is FR1, not conserved → should be MUTABLE
        if "23" in dp:
            assert dp["23"].is_mutable


# ===========================================================================
# DesignPolicy serialisation
# ===========================================================================


class TestDesignPolicySerialization:
    def test_full_roundtrip(self):
        dp = DesignPolicy()
        dp.freeze(["23", "41"])
        dp.restrict("60", {"A", "G", "S"})
        dp.make_mutable(["45", "111A"])

        data = dp.to_dict()
        restored = DesignPolicy.from_dict(data)

        assert restored["23"].is_frozen
        assert restored["41"].is_frozen
        assert restored["60"].is_conservative
        assert restored["60"].allowed_aas == frozenset({"A", "G", "S"})
        assert restored["45"].is_mutable
        assert restored["111A"].is_mutable

    def test_empty_roundtrip(self):
        dp = DesignPolicy()
        assert DesignPolicy.from_dict(dp.to_dict()).policies == {}
