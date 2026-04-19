"""Tests for post-ANARCI numbering validation gates."""

from __future__ import annotations

from tests.conftest import SAMPLE_VHH
from vhh_library.sequence import VHHSequence


def _sequence_with_imgt_mutation(position: int | str, aa: str) -> str:
    source = VHHSequence(SAMPLE_VHH)
    return VHHSequence.mutate(source, position, aa).sequence


def test_valid_vhh_passes_numbering_validation_gates() -> None:
    vhh = VHHSequence(SAMPLE_VHH)
    assert vhh.validation_result["valid"] is True
    assert vhh.validation_result["errors"] == []


def test_cys23_mutation_rejected_in_strict_mode() -> None:
    mutated_sequence = _sequence_with_imgt_mutation(23, "A")
    mutated = VHHSequence(mutated_sequence, strict=True)
    assert mutated.validation_result["valid"] is False
    assert "Missing conserved Cys at IMGT position 23" in mutated.validation_result["errors"]


def test_cys104_mutation_rejected_in_strict_mode() -> None:
    mutated_sequence = _sequence_with_imgt_mutation(104, "A")
    mutated = VHHSequence(mutated_sequence, strict=True)
    assert mutated.validation_result["valid"] is False
    assert "Missing conserved Cys at IMGT position 104" in mutated.validation_result["errors"]


def test_anchor_residue_failures_become_warnings_when_not_strict() -> None:
    cys23_mutated = VHHSequence(_sequence_with_imgt_mutation(23, "A"), strict=False)
    cys104_mutated = VHHSequence(_sequence_with_imgt_mutation(104, "A"), strict=False)

    assert "Missing conserved Cys at IMGT position 23" not in cys23_mutated.validation_result["errors"]
    assert "Missing conserved Cys at IMGT position 23" in cys23_mutated.validation_result["warnings"]
    assert cys23_mutated.validation_result["valid"] is True

    assert "Missing conserved Cys at IMGT position 104" not in cys104_mutated.validation_result["errors"]
    assert "Missing conserved Cys at IMGT position 104" in cys104_mutated.validation_result["warnings"]
    assert cys104_mutated.validation_result["valid"] is True


def test_sequence_reconstruction_matches_known_good_sequence() -> None:
    vhh = VHHSequence(SAMPLE_VHH)
    assert "".join(vhh.imgt_numbered.values()) == SAMPLE_VHH
    assert not any("reconstruct the input sequence" in err for err in vhh.validation_result["errors"])


def test_normal_sequence_has_no_cdr_length_warnings() -> None:
    vhh = VHHSequence(SAMPLE_VHH)
    assert not any("CDR" in warning and "length" in warning for warning in vhh.validation_result["warnings"])


def test_mutate_copies_strict_attribute() -> None:
    source = VHHSequence(SAMPLE_VHH, strict=False)
    mutant = VHHSequence.mutate(source, 1, "E")
    assert mutant.strict is False
