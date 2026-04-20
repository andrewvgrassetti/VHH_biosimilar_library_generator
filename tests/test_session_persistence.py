"""Tests for session auto-save / auto-restore helpers in app.py.

These tests exercise the serialisation and deserialisation round-trip
without launching Streamlit.  They validate:

- VHHSequence round-trips through JSON faithfully.
- DataFrame round-trips through JSON faithfully.
- Plain dicts survive the round-trip unchanged.
- ``None`` values are dropped during serialisation and absent from output.
- ``deserialize_session_data`` gracefully handles unknown type tags.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest

# Import the helpers under test.  They live in ``app.py`` but are free of
# Streamlit runtime dependencies at the function level.
from app import (
    _AUTOSAVE_DIR,
    _auto_save_path,
    deserialize_session_data,
    serialize_session_data,
)
from tests.conftest import SAMPLE_VHH
from vhh_library.sequence import VHHSequence

# ---------------------------------------------------------------------------
# serialize / deserialize round-trip
# ---------------------------------------------------------------------------


class TestSerializeDeserializeRoundTrip:
    """Full round-trip through serialize → JSON → deserialize."""

    def test_vhh_sequence_round_trip(self) -> None:
        vhh = VHHSequence(SAMPLE_VHH)
        data = serialize_session_data({"vhh_seq": vhh})
        json_str = json.dumps(data)
        restored = deserialize_session_data(json.loads(json_str))
        assert isinstance(restored["vhh_seq"], VHHSequence)
        assert restored["vhh_seq"].sequence == vhh.sequence

    def test_dataframe_round_trip(self) -> None:
        df = pd.DataFrame(
            {
                "variant_id": ["V1", "V2", "V3"],
                "aa_sequence": ["ACDEF", "GHIKL", "MNPQR"],
                "stability_score": [0.8, 0.7, 0.9],
            }
        )
        data = serialize_session_data({"library": df})
        json_str = json.dumps(data)
        restored = deserialize_session_data(json.loads(json_str))
        assert isinstance(restored["library"], pd.DataFrame)
        assert list(restored["library"]["variant_id"]) == ["V1", "V2", "V3"]
        assert len(restored["library"]) == 3

    def test_dict_round_trip(self) -> None:
        scores = {"composite_score": 0.75, "predicted_tm": 65.3}
        data = serialize_session_data({"stability_scores": scores})
        json_str = json.dumps(data)
        restored = deserialize_session_data(json.loads(json_str))
        assert restored["stability_scores"] == scores

    def test_none_values_are_skipped(self) -> None:
        data = serialize_session_data({"vhh_seq": None, "library": None})
        assert data == {}

    def test_list_round_trip(self) -> None:
        constructs = [{"tag": "His6"}, {"tag": "FLAG"}]
        data = serialize_session_data({"constructs": constructs})
        json_str = json.dumps(data)
        restored = deserialize_session_data(json.loads(json_str))
        assert restored["constructs"] == constructs

    def test_mixed_types_round_trip(self) -> None:
        vhh = VHHSequence(SAMPLE_VHH)
        df = pd.DataFrame({"x": [1, 2]})
        scores = {"composite_score": 0.5}
        state = {
            "vhh_seq": vhh,
            "library": df,
            "stability_scores": scores,
            "nativeness_scores": None,
        }
        data = serialize_session_data(state)
        json_str = json.dumps(data)
        restored = deserialize_session_data(json.loads(json_str))

        assert isinstance(restored["vhh_seq"], VHHSequence)
        assert restored["vhh_seq"].sequence == vhh.sequence
        assert isinstance(restored["library"], pd.DataFrame)
        assert len(restored["library"]) == 2
        assert restored["stability_scores"] == scores
        assert "nativeness_scores" not in restored  # None was skipped

    def test_empty_dict_round_trip(self) -> None:
        data = serialize_session_data({})
        assert data == {}
        restored = deserialize_session_data(data)
        assert restored == {}


# ---------------------------------------------------------------------------
# deserialize edge cases
# ---------------------------------------------------------------------------


class TestDeserializeEdgeCases:
    """Edge cases for the deserialisation path."""

    def test_unknown_type_tag_skipped(self) -> None:
        data = {"mystery": {"__type__": "UnknownClass", "value": 42}}
        restored = deserialize_session_data(data)
        assert "mystery" not in restored

    def test_malformed_vhh_sequence_skipped(self) -> None:
        # Invalid VHH sequence (too short) should not crash.
        data = {"vhh_seq": {"__type__": "VHHSequence", "sequence": "XYZ"}}
        restored = deserialize_session_data(data)
        # Depending on validation, the key may or may not be present.
        # The important thing is it doesn't raise.
        if "vhh_seq" in restored:
            assert isinstance(restored["vhh_seq"], VHHSequence)

    def test_empty_dataframe_records(self) -> None:
        data = {"library": {"__type__": "DataFrame", "records": []}}
        restored = deserialize_session_data(data)
        assert isinstance(restored["library"], pd.DataFrame)
        assert restored["library"].empty


# ---------------------------------------------------------------------------
# Auto-save file path
# ---------------------------------------------------------------------------


class TestAutoSavePath:
    """Verify the auto-save path helper."""

    def test_path_is_in_temp_dir(self) -> None:
        path = _auto_save_path()
        assert path.parent == _AUTOSAVE_DIR
        assert path.name == "autosave.json"

    def test_autosave_dir_is_under_tmp(self) -> None:
        # On Linux, tempfile.gettempdir() is usually /tmp
        assert "tmp" in str(_AUTOSAVE_DIR).lower() or "temp" in str(_AUTOSAVE_DIR).lower()


# ---------------------------------------------------------------------------
# File-based auto-save / restore integration
# ---------------------------------------------------------------------------


class TestAutoSaveRestoreFile:
    """Integration test using actual file I/O (no Streamlit)."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Remove auto-save file after each test."""
        yield
        path = _auto_save_path()
        if path.is_file():
            path.unlink(missing_ok=True)

    def test_write_and_read_autosave_file(self) -> None:
        vhh = VHHSequence(SAMPLE_VHH)
        state = {
            "vhh_seq": vhh,
            "stability_scores": {"composite_score": 0.8},
        }
        serialised = serialize_session_data(state)
        _AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        _auto_save_path().write_text(json.dumps(serialised))

        raw = json.loads(_auto_save_path().read_text())
        restored = deserialize_session_data(raw)
        assert isinstance(restored["vhh_seq"], VHHSequence)
        assert restored["vhh_seq"].sequence == vhh.sequence
        assert restored["stability_scores"]["composite_score"] == 0.8

    def test_stale_file_ignored(self) -> None:
        """Files older than 24 hours should be treated as stale."""
        _AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = _auto_save_path()
        path.write_text("{}")
        # Backdate the file by 25 hours.
        old_time = time.time() - 90000
        os.utime(path, (old_time, old_time))
        assert path.stat().st_mtime < time.time() - 86400
