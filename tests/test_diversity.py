"""Unit tests for vhh_library.diversity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vhh_library.diversity import (
    compute_umap_embedding,
    encode_mutation_matrix,
    mutation_frequency_matrix,
    pairwise_cooccurrence_matrix,
)


def _make_library(mutations: list[str], n_mutations: list[int] | None = None) -> pd.DataFrame:
    """Helper to build a minimal library DataFrame."""
    n = len(mutations)
    if n_mutations is None:
        n_mutations = [m.count(",") + 1 if m else 0 for m in mutations]
    return pd.DataFrame(
        {
            "variant_id": list(range(1, n + 1)),
            "mutations": mutations,
            "n_mutations": n_mutations,
            "aa_sequence": ["EVQLVESGGGLVQPGGSLRLSCAAS"] * n,
            "combined_score": [0.9 - 0.1 * i for i in range(n)],
            "stability_score": [0.8] * n,
            "nativeness_score": [0.7] * n,
        }
    )


class TestEncodeMutationMatrix:
    """Tests for encode_mutation_matrix."""

    def test_basic_shape_and_values(self):
        lib = _make_library(["A10S", "A10S, G42K", "G42K", ""])
        mat, positions = encode_mutation_matrix(lib, wt_sequence="")
        assert mat.shape == (4, 2)
        assert positions == ["10", "42"]
        # Row 0: only pos 10 mutated
        assert mat[0, 0] != 0  # S encoding
        assert mat[0, 1] == 0
        # Row 1: both mutated
        assert mat[1, 0] != 0
        assert mat[1, 1] != 0
        # Row 2: only pos 42
        assert mat[2, 0] == 0
        assert mat[2, 1] != 0
        # Row 3: no mutations
        assert mat[3, 0] == 0
        assert mat[3, 1] == 0

    def test_insertion_code_positions(self):
        lib = _make_library(["A111AS"])
        mat, positions = encode_mutation_matrix(lib, wt_sequence="")
        assert positions == ["111A"]
        assert mat.shape == (1, 1)
        assert mat[0, 0] != 0

    def test_empty_library(self):
        lib = _make_library([])
        mat, positions = encode_mutation_matrix(lib, wt_sequence="")
        assert mat.shape == (0, 0)
        assert positions == []

    def test_no_mutations(self):
        lib = _make_library(["", ""])
        mat, positions = encode_mutation_matrix(lib, wt_sequence="")
        assert mat.shape == (2, 0)
        assert positions == []

    def test_wt_position_is_zero(self):
        lib = _make_library(["A10S", "A10G"])
        mat, positions = encode_mutation_matrix(lib, wt_sequence="")
        assert positions == ["10"]
        # Both should be non-zero (not wild-type)
        assert mat[0, 0] != 0
        assert mat[1, 0] != 0
        # And they should differ (S != G)
        assert mat[0, 0] != mat[1, 0]


class TestMutationFrequencyMatrix:
    """Tests for mutation_frequency_matrix."""

    def test_basic_frequencies(self):
        lib = _make_library(["A10S", "A10S", "A10G", "A10S"])
        freq = mutation_frequency_matrix(lib)
        assert freq.shape[0] == 1  # 1 position
        assert freq.shape[1] == 20  # 20 amino acids
        # S appears 3 times in 4 variants
        assert freq.loc["10", "S"] == pytest.approx(3 / 4)
        # G appears 1 time in 4 variants
        assert freq.loc["10", "G"] == pytest.approx(1 / 4)

    def test_top_n_filter(self):
        lib = _make_library(["A10S", "A10G", "A10L", "A10F"])
        # combined_score: 0.9, 0.8, 0.7, 0.6
        freq = mutation_frequency_matrix(lib, top_n=2)
        # Only top 2 variants (S and G)
        assert freq.loc["10", "S"] == pytest.approx(1 / 2)
        assert freq.loc["10", "G"] == pytest.approx(1 / 2)

    def test_empty_mutations(self):
        lib = _make_library(["", ""])
        freq = mutation_frequency_matrix(lib)
        assert freq.empty


class TestPairwiseCooccurrenceMatrix:
    """Tests for pairwise_cooccurrence_matrix."""

    def test_symmetry(self):
        lib = _make_library(["A10S, G42K", "A10S", "G42K"])
        cooc = pairwise_cooccurrence_matrix(lib)
        assert cooc.shape[0] == cooc.shape[1]
        np.testing.assert_array_equal(cooc.values, cooc.values.T)

    def test_known_counts(self):
        lib = _make_library(["A10S, G42K", "A10S, G42K", "A10S"])
        cooc = pairwise_cooccurrence_matrix(lib)
        # Pos 10 mutated in 3 variants (diagonal)
        assert cooc.loc["10", "10"] == 3
        # Pos 42 mutated in 2 variants (diagonal)
        assert cooc.loc["42", "42"] == 2
        # Both co-occur in 2 variants
        assert cooc.loc["10", "42"] == 2
        assert cooc.loc["42", "10"] == 2

    def test_top_n_filter(self):
        lib = _make_library(["A10S, G42K", "A10S", "G42K", "A10S, G42K"])
        # combined_score: 0.9, 0.8, 0.7, 0.6
        cooc = pairwise_cooccurrence_matrix(lib, top_n=2)
        # Top 2: variant 1 (A10S, G42K) and variant 2 (A10S)
        assert cooc.loc["10", "10"] == 2
        assert cooc.loc["42", "42"] == 1
        assert cooc.loc["10", "42"] == 1

    def test_empty_mutations(self):
        lib = _make_library(["", ""])
        cooc = pairwise_cooccurrence_matrix(lib)
        assert cooc.empty


class TestComputeUmapEmbedding:
    """Tests for compute_umap_embedding."""

    @pytest.mark.slow
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        matrix = rng.integers(0, 5, size=(10, 4), dtype=np.int8)
        embedding = compute_umap_embedding(matrix, n_neighbors=3)
        assert embedding.shape == (10, 2)
        assert embedding.dtype == np.float64
