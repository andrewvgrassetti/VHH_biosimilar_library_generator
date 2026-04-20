"""Tests for the ANARCI / BioPython compatibility patch in vhh_library.numbering."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest import mock

import pytest

import vhh_library.numbering as numbering_mod


@pytest.fixture(autouse=True)
def _reset_patch_flag():
    """Reset the module-level _PATCHED flag before each test."""
    original = numbering_mod._PATCHED
    numbering_mod._PATCHED = False
    yield
    numbering_mod._PATCHED = original


# ---------------------------------------------------------------------------
# Helpers -- lightweight HSP / fragment stubs
# ---------------------------------------------------------------------------


def _make_fragment(*, query_start=10, query_end=120):
    """Return a minimal HSP-fragment-like object."""
    return SimpleNamespace(query_start=query_start, query_end=query_end)


def _make_hsp(
    *,
    query_start=10,
    query_end=120,
    env_start=5,
    env_end=125,
    fragments=None,
):
    """Return a minimal HSP-like object with one fragment by default."""
    frag = _make_fragment(query_start=query_start, query_end=query_end)
    return SimpleNamespace(
        query_start=query_start,
        query_end=query_end,
        env_start=env_start,
        env_end=env_end,
        fragments=fragments if fragments is not None else [frag],
    )


def _make_query(hsps):
    """Return a minimal query-like object."""
    return SimpleNamespace(hsps=hsps)


# ---------------------------------------------------------------------------
# _patched_domains_are_same
# ---------------------------------------------------------------------------


class TestPatchedDomainsAreSame:
    """Verify the _domains_are_same patch handles None coordinates."""

    def test_both_none_query_start(self):
        numbering_mod._apply_anarci_compat_patch()
        anarci_mod = importlib.import_module("anarci.anarci")

        dom1 = SimpleNamespace(query_start=None, query_end=None, env_start=10, env_end=100)
        dom2 = SimpleNamespace(query_start=None, query_end=None, env_start=50, env_end=120)
        # Should not raise -- falls back to env_start / env_end
        result = anarci_mod._domains_are_same(dom1, dom2)
        assert isinstance(result, bool)

    def test_normal_values_still_work(self):
        numbering_mod._apply_anarci_compat_patch()
        anarci_mod = importlib.import_module("anarci.anarci")

        dom1 = SimpleNamespace(query_start=10, query_end=100)
        dom2 = SimpleNamespace(query_start=50, query_end=120)
        result = anarci_mod._domains_are_same(dom1, dom2)
        assert result is True  # 50 < 100

    def test_non_overlapping_returns_false(self):
        numbering_mod._apply_anarci_compat_patch()
        anarci_mod = importlib.import_module("anarci.anarci")

        dom1 = SimpleNamespace(query_start=10, query_end=50)
        dom2 = SimpleNamespace(query_start=60, query_end=120)
        result = anarci_mod._domains_are_same(dom1, dom2)
        assert result is False  # 60 >= 50


# ---------------------------------------------------------------------------
# _patched_parse -- query coordinate fix
# ---------------------------------------------------------------------------


class TestPatchedParseQueryCoords:
    """Verify the _parse_hmmer_query patch back-fills query_start/query_end."""

    def test_query_none_backfilled_from_env(self):
        """When query_start is None, fragments should inherit env_start/env_end."""
        numbering_mod._apply_anarci_compat_patch()
        anarci_mod = importlib.import_module("anarci.anarci")

        frag = _make_fragment(query_start=None, query_end=None)
        hsp = _make_hsp(
            query_start=None,
            query_end=None,
            env_start=7,
            env_end=130,
            fragments=[frag],
        )
        query = _make_query([hsp])

        def capturing_original(query, bit_score_threshold=80, hmmer_species=None):
            return []

        # Re-apply the patch with a controlled original
        numbering_mod._PATCHED = False
        with mock.patch.object(anarci_mod, "_parse_hmmer_query", capturing_original):
            numbering_mod._apply_anarci_compat_patch()
            anarci_mod._parse_hmmer_query(query)

        assert frag.query_start == 7
        assert frag.query_end == 130

    def test_query_present_not_overwritten(self):
        """When query_start is already set, it should not be overwritten."""
        numbering_mod._apply_anarci_compat_patch()
        anarci_mod = importlib.import_module("anarci.anarci")

        frag = _make_fragment(query_start=10, query_end=120)
        hsp = _make_hsp(
            query_start=10,
            query_end=120,
            env_start=3,
            env_end=128,
            fragments=[frag],
        )
        query = _make_query([hsp])

        def capturing_original(query, bit_score_threshold=80, hmmer_species=None):
            return []

        numbering_mod._PATCHED = False
        with mock.patch.object(anarci_mod, "_parse_hmmer_query", capturing_original):
            numbering_mod._apply_anarci_compat_patch()
            anarci_mod._parse_hmmer_query(query)

        # Original values preserved
        assert frag.query_start == 10
        assert frag.query_end == 120


# ---------------------------------------------------------------------------
# _patched_hmm_alignment_to_states
# ---------------------------------------------------------------------------


class TestPatchedHmmAlignmentToStates:
    """Verify the _hmm_alignment_to_states patch handles TypeError from None."""

    def test_typeerror_retried_with_seq_length_zero(self):
        """If the original raises TypeError, patch retries with seq_length=0."""
        numbering_mod._apply_anarci_compat_patch()
        anarci_mod = importlib.import_module("anarci.anarci")

        calls: list[tuple] = []
        sentinel = [("state1", 1, "A")]

        def tracking_original(hsp, n, seq_length):
            calls.append((hsp, n, seq_length))
            if seq_length > 0:
                raise TypeError("'<' not supported between instances of 'int' and 'NoneType'")
            return sentinel

        numbering_mod._PATCHED = False
        with mock.patch.object(anarci_mod, "_hmm_alignment_to_states", tracking_original):
            numbering_mod._apply_anarci_compat_patch()
            result = anarci_mod._hmm_alignment_to_states("hsp", 1, 118)

        assert result is sentinel
        assert len(calls) == 2
        assert calls[0] == ("hsp", 1, 118)  # first try with original seq_length
        assert calls[1] == ("hsp", 1, 0)  # retry with seq_length=0

    def test_non_typeerror_propagates(self):
        """Non-TypeError exceptions should still propagate."""
        numbering_mod._apply_anarci_compat_patch()
        anarci_mod = importlib.import_module("anarci.anarci")

        def raising_original(hsp, n, seq_length):
            raise ValueError("some other error")

        numbering_mod._PATCHED = False
        with mock.patch.object(anarci_mod, "_hmm_alignment_to_states", raising_original):
            numbering_mod._apply_anarci_compat_patch()
            with pytest.raises(ValueError, match="some other error"):
                anarci_mod._hmm_alignment_to_states("hsp", 1, 118)

    def test_normal_return_passes_through(self):
        """When the original succeeds, result is passed through unchanged."""
        numbering_mod._apply_anarci_compat_patch()
        anarci_mod = importlib.import_module("anarci.anarci")

        sentinel = [("state1", 1, "A")]

        def normal_original(hsp, n, seq_length):
            return sentinel

        numbering_mod._PATCHED = False
        with mock.patch.object(anarci_mod, "_hmm_alignment_to_states", normal_original):
            numbering_mod._apply_anarci_compat_patch()
            result = anarci_mod._hmm_alignment_to_states("hsp", 1, 118)

        assert result is sentinel


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestPatchIdempotency:
    """Verify the _PATCHED guard prevents double-patching."""

    def test_second_call_is_noop(self):
        numbering_mod._apply_anarci_compat_patch()
        assert numbering_mod._PATCHED is True

        # Capture the current patched function references
        anarci_mod = importlib.import_module("anarci.anarci")
        domains_fn = anarci_mod._domains_are_same
        parse_fn = anarci_mod._parse_hmmer_query
        hmm_fn = anarci_mod._hmm_alignment_to_states

        # Second call should be a no-op
        numbering_mod._apply_anarci_compat_patch()
        assert anarci_mod._domains_are_same is domains_fn
        assert anarci_mod._parse_hmmer_query is parse_fn
        assert anarci_mod._hmm_alignment_to_states is hmm_fn


# ---------------------------------------------------------------------------
# NanoMelt _ensure_backend calls the patch
# ---------------------------------------------------------------------------


class TestNanoMeltCallsPatch:
    """Verify NanoMeltPredictor._ensure_backend applies the ANARCI compat patch."""

    def test_ensure_backend_calls_compat_patch(self):
        """_ensure_backend must call _apply_anarci_compat_patch before importing NanoMeltPredPipe."""
        from unittest.mock import MagicMock

        call_order: list[str] = []

        def tracking_patch():
            call_order.append("patch")

        mock_pred_pipe = MagicMock()

        with (
            mock.patch("vhh_library.predictors.nanomelt.NANOMELT_AVAILABLE", True),
            mock.patch("vhh_library.numbering._apply_anarci_compat_patch", tracking_patch),
        ):
            from vhh_library.predictors.nanomelt import NanoMeltPredictor

            pred = NanoMeltPredictor(device="cpu")
            with mock.patch.dict(
                "sys.modules",
                {
                    "nanomelt": MagicMock(),
                    "nanomelt.predict": MagicMock(NanoMeltPredPipe=mock_pred_pipe),
                },
            ):
                pred._ensure_backend()

        assert "patch" in call_order
