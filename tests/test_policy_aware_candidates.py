"""Tests for policy-aware candidate generation and multi-mutant rescoring.

Validates that:
* Frozen positions never emit candidates.
* Conservative positions enforce allowed_aas.
* Candidate metadata fields are fully populated.
* Multi-mutant rescoring uses full-sequence inference (not delta-summing).
* Insertion-coded positions are preserved as strings.
* Liability flags are detected and recorded.
* ESM-2 prior scoring is optional and correctly populated when provided.
"""

from __future__ import annotations

import pytest

from vhh_library.mutation_engine import (
    MutationCandidate,
    MutationEngine,
    _detect_new_ptm_liabilities,
)
from vhh_library.position_policy import (
    DesignPolicy,
    PositionClass,
    PositionPolicy,
    from_vhh_sequence,
)
from vhh_library.predictors.base import Predictor
from vhh_library.sequence import VHHSequence
from vhh_library.stability import StabilityScorer
from vhh_library.utils import AMINO_ACIDS

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


# ---------------------------------------------------------------------------
# Stub predictors that return deterministic scores
# ---------------------------------------------------------------------------


class _StubAbNatiV(Predictor):
    """Deterministic AbNatiV stub."""

    @property
    def name(self) -> str:
        return "abnativ"

    def score_sequence(self, sequence: VHHSequence) -> dict[str, float]:
        # Use length % 20 to produce a repeatable score.
        raw = (len(sequence.sequence) % 20) / 20.0
        return {"composite_score": 0.5 + raw * 0.4}


class _StubNanoMelt(Predictor):
    """Deterministic NanoMelt stub that returns a fake Tm."""

    @property
    def name(self) -> str:
        return "nanomelt"

    def score_sequence(self, sequence: VHHSequence) -> dict[str, float]:
        # Use length modulo to produce a Tm in [60, 80].
        tm = 60.0 + (len(sequence.sequence) % 20)
        return {"composite_score": 0.7, "nanomelt_tm": tm}


class _StubESM2(Predictor):
    """Deterministic ESM-2 stub."""

    @property
    def name(self) -> str:
        return "esm2_prior"

    def score_sequence(self, sequence: VHHSequence) -> dict[str, float]:
        pll = -100.0 - (len(sequence.sequence) % 10)
        return {"composite_score": 0.65, "esm2_pll": pll}


class _MockNativenessScorer:
    """Deterministic mock for the legacy NativenessScorer interface."""

    def score(self, vhh: VHHSequence) -> dict:
        return {"composite_score": 0.7}

    def predict_mutation_effect(self, vhh: VHHSequence, position: int | str, new_aa: str) -> float:
        return 0.01

    def score_batch(self, sequences: list[str]) -> list[float]:
        return [0.7] * len(sequences)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vhh() -> VHHSequence:
    return VHHSequence(SAMPLE_VHH)


@pytest.fixture(scope="module")
def engine() -> MutationEngine:
    return MutationEngine(
        stability_scorer=StabilityScorer(),
        nativeness_scorer=_MockNativenessScorer(),
    )


@pytest.fixture
def abnativ_pred() -> _StubAbNatiV:
    return _StubAbNatiV()


@pytest.fixture
def nanomelt_pred() -> _StubNanoMelt:
    return _StubNanoMelt()


@pytest.fixture
def esm2_pred() -> _StubESM2:
    return _StubESM2()


# ===========================================================================
# MutationCandidate dataclass
# ===========================================================================


class TestMutationCandidate:
    def test_creation(self) -> None:
        mc = MutationCandidate(
            imgt_pos="27",
            original_aa="A",
            suggested_aa="G",
            position_class="mutable",
            reasons=["Test reason"],
        )
        assert mc.imgt_pos == "27"
        assert mc.original_aa == "A"
        assert mc.suggested_aa == "G"
        assert mc.position_class == "mutable"
        assert mc.abnativ_score is None
        assert mc.liability_flags == []

    def test_to_dict(self) -> None:
        mc = MutationCandidate(
            imgt_pos="111A",
            original_aa="G",
            suggested_aa="A",
            position_class="conservative",
            reasons=["Conservative: restricted"],
            abnativ_score=0.85,
            abnativ_delta=0.05,
            nanomelt_tm=72.3,
            nanomelt_delta_tm=2.1,
            esm2_prior_score=0.65,
            esm2_pll=-110.0,
            liability_flags=["deamidation"],
        )
        d = mc.to_dict()
        assert d["imgt_pos"] == "111A"
        assert d["position_class"] == "conservative"
        assert d["abnativ_score"] == pytest.approx(0.85)
        assert d["nanomelt_tm"] == pytest.approx(72.3)
        assert d["esm2_pll"] == pytest.approx(-110.0)
        assert "deamidation" in d["liability_flags"]

    def test_to_dict_preserves_insertion_codes(self) -> None:
        mc = MutationCandidate(
            imgt_pos="111B",
            original_aa="T",
            suggested_aa="S",
            position_class="mutable",
        )
        assert mc.to_dict()["imgt_pos"] == "111B"


# ===========================================================================
# _detect_new_ptm_liabilities
# ===========================================================================


class TestDetectNewPTMLiabilities:
    def test_no_new_liability(self) -> None:
        parent = "AAAAAAAAAA"
        mutant = "AAAAAAAAAA"
        assert _detect_new_ptm_liabilities(parent, mutant, 5) == []

    def test_introduces_deamidation(self) -> None:
        # N followed by G/S/H triggers deamidation.
        # NG at pos 4-5 would trigger deamidation:
        parent_without_ng = "AAAAAGAAA"
        mutant_with_ng = "AAAANGAAA"
        flags = _detect_new_ptm_liabilities(parent_without_ng, mutant_with_ng, 4)
        assert "deamidation" in flags

    def test_returns_specific_categories(self) -> None:
        # DG triggers isomerization.
        parent_without_dg = "AAAAAGAAA"
        mutant_with_dg = "AAAADGAAA"
        flags = _detect_new_ptm_liabilities(parent_without_dg, mutant_with_dg, 4)
        assert "isomerization" in flags


# ===========================================================================
# Frozen positions never emit candidates
# ===========================================================================


class TestFrozenPositionsExcluded:
    def test_frozen_positions_produce_no_candidates(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """All positions frozen → zero candidates."""
        policy = DesignPolicy()
        for pos_key in vhh.imgt_numbered:
            policy.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)

        candidates = engine.generate_policy_aware_candidates(
            vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
        )
        assert len(candidates) == 0

    def test_cdr_default_frozen_no_candidates(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """Default policy freezes CDRs — CDR positions should not appear in candidates."""
        policy = from_vhh_sequence(vhh)
        candidates = engine.generate_policy_aware_candidates(
            vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
        )
        cdr_positions = vhh.cdr_positions
        for c in candidates:
            assert c.imgt_pos not in cdr_positions, f"CDR position {c.imgt_pos} should be frozen"

    def test_conserved_positions_frozen(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """Conserved positions (23, 41, 104) should not emit candidates in default policy."""
        policy = from_vhh_sequence(vhh)
        candidates = engine.generate_policy_aware_candidates(
            vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
        )
        candidate_positions = {c.imgt_pos for c in candidates}
        assert "23" not in candidate_positions
        assert "41" not in candidate_positions
        assert "104" not in candidate_positions


# ===========================================================================
# Conservative positions enforce allowed_aas
# ===========================================================================


class TestConservativePositions:
    def test_conservative_restricts_substitutions(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """Conservative positions should only produce candidates in allowed_aas."""
        policy = DesignPolicy()
        # Freeze everything, then make one position conservative.
        for pos_key in vhh.imgt_numbered:
            policy.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)

        # Pick a framework position and set it conservative with restricted AAs.
        target_pos = "5"
        if target_pos in vhh.imgt_numbered:
            allowed = frozenset({"A", "G", "V"})
            policy.policies[target_pos] = PositionPolicy(target_pos, PositionClass.CONSERVATIVE, allowed)

            candidates = engine.generate_policy_aware_candidates(
                vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
            )

            for c in candidates:
                assert c.imgt_pos == target_pos
                assert c.suggested_aa in allowed, (
                    f"Conservative position {target_pos}: {c.suggested_aa} not in {allowed}"
                )
                assert c.position_class == "conservative"

    def test_conservative_excludes_original_aa(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """Candidates should not include the original amino acid."""
        policy = DesignPolicy()
        for pos_key in vhh.imgt_numbered:
            policy.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)

        target_pos = "5"
        original_aa = vhh.imgt_numbered.get(target_pos)
        if original_aa is not None:
            # Include the original AA in allowed_aas — should still not appear.
            allowed = frozenset({"A", "G", "V", original_aa})
            policy.policies[target_pos] = PositionPolicy(target_pos, PositionClass.CONSERVATIVE, allowed)

            candidates = engine.generate_policy_aware_candidates(
                vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
            )
            for c in candidates:
                assert c.suggested_aa != original_aa


# ===========================================================================
# Candidate metadata is fully populated
# ===========================================================================


class TestCandidateMetadata:
    def test_all_metadata_fields_present(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """Every candidate should have all metadata fields populated."""
        policy = DesignPolicy()
        # Make a single position mutable.
        for pos_key in vhh.imgt_numbered:
            policy.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)
        target_pos = "5"
        if target_pos in vhh.imgt_numbered:
            policy.policies[target_pos] = PositionPolicy(target_pos, PositionClass.MUTABLE)

            candidates = engine.generate_policy_aware_candidates(
                vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
            )
            assert len(candidates) > 0
            for c in candidates:
                assert c.imgt_pos == target_pos
                assert c.original_aa == vhh.imgt_numbered[target_pos]
                assert c.suggested_aa in AMINO_ACIDS
                assert c.suggested_aa != c.original_aa
                assert c.position_class == "mutable"
                assert len(c.reasons) > 0
                # AbNatiV fields populated.
                assert c.abnativ_score is not None
                assert c.abnativ_delta is not None
                # NanoMelt fields populated.
                assert c.nanomelt_tm is not None
                assert c.nanomelt_delta_tm is not None
                # ESM-2 fields remain None when predictor not provided.
                assert c.esm2_prior_score is None
                assert c.esm2_pll is None
                # Liability flags is a list (may be empty).
                assert isinstance(c.liability_flags, list)

    def test_esm2_metadata_when_provided(
        self,
        engine: MutationEngine,
        vhh: VHHSequence,
        abnativ_pred: _StubAbNatiV,
        nanomelt_pred: _StubNanoMelt,
        esm2_pred: _StubESM2,
    ) -> None:
        """When ESM-2 predictor is provided, esm2 fields should be populated."""
        policy = DesignPolicy()
        for pos_key in vhh.imgt_numbered:
            policy.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)
        target_pos = "5"
        if target_pos in vhh.imgt_numbered:
            policy.policies[target_pos] = PositionPolicy(target_pos, PositionClass.MUTABLE)

            candidates = engine.generate_policy_aware_candidates(
                vhh,
                policy,
                abnativ_predictor=abnativ_pred,
                nanomelt_predictor=nanomelt_pred,
                esm2_predictor=esm2_pred,
            )
            assert len(candidates) > 0
            for c in candidates:
                assert c.esm2_prior_score is not None
                assert c.esm2_pll is not None

    def test_to_dict_converts_cleanly(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """to_dict() should produce a flat dict suitable for DataFrame construction."""
        policy = DesignPolicy()
        for pos_key in vhh.imgt_numbered:
            policy.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)
        target_pos = "5"
        if target_pos in vhh.imgt_numbered:
            policy.policies[target_pos] = PositionPolicy(target_pos, PositionClass.MUTABLE)

            candidates = engine.generate_policy_aware_candidates(
                vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
            )
            for c in candidates:
                d = c.to_dict()
                assert isinstance(d, dict)
                expected_keys = {
                    "imgt_pos",
                    "original_aa",
                    "suggested_aa",
                    "position_class",
                    "reasons",
                    "abnativ_score",
                    "abnativ_delta",
                    "nanomelt_tm",
                    "nanomelt_delta_tm",
                    "esm2_prior_score",
                    "esm2_pll",
                    "liability_flags",
                }
                assert set(d.keys()) == expected_keys

    def test_insertion_coded_positions_not_collapsed(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """Insertion-coded positions like '111A' must not be collapsed to integers."""
        # Find any insertion-coded positions in the sequence.
        insertion_positions = [k for k in vhh.imgt_numbered if not k.isdigit()]
        if not insertion_positions:
            pytest.skip("No insertion-coded positions in sample sequence")

        policy = DesignPolicy()
        for pos_key in vhh.imgt_numbered:
            policy.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)
        # Make one insertion position mutable.
        ins_pos = insertion_positions[0]
        policy.policies[ins_pos] = PositionPolicy(ins_pos, PositionClass.MUTABLE)

        candidates = engine.generate_policy_aware_candidates(
            vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
        )
        for c in candidates:
            assert c.imgt_pos == ins_pos
            # Must be the original string, not an integer.
            assert isinstance(c.imgt_pos, str)
            assert not c.imgt_pos.isdigit()  # has insertion code


# ===========================================================================
# Excluded target amino acids
# ===========================================================================


class TestExcludedTargetAAs:
    def test_excluded_aas_not_in_candidates(
        self, engine: MutationEngine, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt
    ) -> None:
        """Globally excluded AAs should not appear in candidates."""
        policy = from_vhh_sequence(vhh)
        candidates = engine.generate_policy_aware_candidates(
            vhh,
            policy,
            abnativ_predictor=abnativ_pred,
            nanomelt_predictor=nanomelt_pred,
            excluded_target_aas={"C", "M"},
        )
        for c in candidates:
            assert c.suggested_aa not in {"C", "M"}


# ===========================================================================
# Multi-mutant rescoring path
# ===========================================================================


class TestMultiMutantRescoring:
    def test_rescore_returns_all_keys(
        self, vhh: VHHSequence, abnativ_pred: _StubAbNatiV, nanomelt_pred: _StubNanoMelt, esm2_pred: _StubESM2
    ) -> None:
        """Rescoring a multi-mutant should return all expected keys."""
        mutations = [("5", "A"), ("9", "G")]
        result = MutationEngine.rescore_multi_mutant(
            vhh,
            mutations,
            abnativ_predictor=abnativ_pred,
            nanomelt_predictor=nanomelt_pred,
            esm2_predictor=esm2_pred,
        )
        assert "abnativ_score" in result
        assert "nanomelt_tm" in result
        assert "esm2_prior_score" in result
        assert "esm2_pll" in result
        assert result["abnativ_score"] is not None
        assert result["nanomelt_tm"] is not None
        assert result["esm2_prior_score"] is not None
        assert result["esm2_pll"] is not None

    def test_rescore_none_predictors(self, vhh: VHHSequence) -> None:
        """With no predictors, all values should be None."""
        mutations = [("5", "A")]
        result = MutationEngine.rescore_multi_mutant(vhh, mutations)
        assert result["abnativ_score"] is None
        assert result["nanomelt_tm"] is None
        assert result["esm2_prior_score"] is None
        assert result["esm2_pll"] is None

    def test_rescore_applies_all_mutations(self, vhh: VHHSequence) -> None:
        """Verify that multi-mutant rescoring applies all mutations, not just the first."""
        call_log: list[str] = []

        class _LoggingPredictor(Predictor):
            @property
            def name(self) -> str:
                return "logger"

            def score_sequence(self, sequence: VHHSequence) -> dict[str, float]:
                call_log.append(sequence.sequence)
                return {"composite_score": 0.5}

        pred = _LoggingPredictor()
        mutations = [("5", "A"), ("9", "G")]
        MutationEngine.rescore_multi_mutant(vhh, mutations, abnativ_predictor=pred)

        # The predictor should have been called once with the doubly-mutated sequence.
        assert len(call_log) == 1
        scored_seq = call_log[0]
        # Both mutations should be applied.
        pos5_idx = vhh._pos_to_seq_idx.get("5")
        pos9_idx = vhh._pos_to_seq_idx.get("9")
        if pos5_idx is not None:
            assert scored_seq[pos5_idx] == "A"
        if pos9_idx is not None:
            assert scored_seq[pos9_idx] == "G"

    def test_rescore_does_not_sum_deltas(self, vhh: VHHSequence) -> None:
        """Ensure rescoring uses full-sequence inference, not delta summing.

        We verify this by checking the predictor receives the full mutant
        sequence (not the wild-type) and is called exactly once.
        """
        calls: list[VHHSequence] = []

        class _TrackingPredictor(Predictor):
            @property
            def name(self) -> str:
                return "tracker"

            def score_sequence(self, sequence: VHHSequence) -> dict[str, float]:
                calls.append(sequence)
                return {"composite_score": 0.6, "nanomelt_tm": 70.0}

        pred = _TrackingPredictor()
        mutations = [("5", "A"), ("9", "G")]
        MutationEngine.rescore_multi_mutant(vhh, mutations, nanomelt_predictor=pred)

        # Exactly one call with the multi-mutant sequence.
        assert len(calls) == 1
        mutant_seq = calls[0]
        # The sequence should differ from wild-type at the mutated positions.
        assert mutant_seq.sequence != vhh.sequence


# ===========================================================================
# Compatibility: existing rank_single_mutations still works
# ===========================================================================


class TestCompatibilityMode:
    def test_rank_single_mutations_unchanged(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        """The old rank_single_mutations path should still work identically."""
        df = engine.rank_single_mutations(vhh)
        assert "position" in df.columns
        assert "combined_score" in df.columns
        assert "delta_stability" in df.columns
        assert "delta_nativeness" in df.columns
        if not df.empty:
            # CDR positions should not appear.
            cdr_positions = vhh.cdr_positions
            for _, row in df.iterrows():
                assert str(row["imgt_pos"]) not in cdr_positions

    def test_both_methods_produce_results(
        self,
        engine: MutationEngine,
        vhh: VHHSequence,
        abnativ_pred: _StubAbNatiV,
        nanomelt_pred: _StubNanoMelt,
    ) -> None:
        """Both old and new paths should produce non-empty results."""
        # Old path
        df = engine.rank_single_mutations(vhh)
        assert not df.empty

        # New path
        policy = from_vhh_sequence(vhh)
        candidates = engine.generate_policy_aware_candidates(
            vhh, policy, abnativ_predictor=abnativ_pred, nanomelt_predictor=nanomelt_pred
        )
        assert len(candidates) > 0


# ===========================================================================
# No predictors provided — candidates still generated (scores stay None)
# ===========================================================================


class TestNoPredictors:
    def test_candidates_without_predictors(self, engine: MutationEngine, vhh: VHHSequence) -> None:
        """Without any predictor, candidates should still be generated with None scores."""
        policy = DesignPolicy()
        for pos_key in vhh.imgt_numbered:
            policy.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)
        target_pos = "5"
        if target_pos in vhh.imgt_numbered:
            policy.policies[target_pos] = PositionPolicy(target_pos, PositionClass.MUTABLE)
            candidates = engine.generate_policy_aware_candidates(vhh, policy)
            assert len(candidates) > 0
            for c in candidates:
                assert c.abnativ_score is None
                assert c.nanomelt_tm is None
                assert c.esm2_prior_score is None
