"""Tests for vhh_library.library_manager – LibraryManager class."""

from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest

from vhh_library.library_manager import LibraryManager


@pytest.fixture
def manager() -> LibraryManager:
    return LibraryManager()


_CLEANUP_DIRS: list[str] = []


@pytest.fixture(autouse=True)
def _cleanup():
    """Clean up any directories created during tests."""
    yield
    for d in _CLEANUP_DIRS:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    _CLEANUP_DIRS.clear()


class TestSessionID:
    def test_session_id_format(self, manager: LibraryManager) -> None:
        sid = manager.session_id
        assert len(sid) == 15
        assert "_" in sid


class TestVariantID:
    def test_create_variant_id(self, manager: LibraryManager) -> None:
        vid = manager.create_variant_id(1)
        assert vid.startswith("VHH-")


class TestSaveLoad:
    def test_save_load_roundtrip(self, manager: LibraryManager) -> None:
        out_dir = f"test_sessions_{manager.session_id}"
        _CLEANUP_DIRS.append(out_dir)
        data = {"key": "value", "count": 42}
        paths = manager.save_session(data, output_dir=out_dir)
        loaded = manager.load_session(paths["json"])
        assert loaded["key"] == "value"
        assert loaded["count"] == 42


class TestExportFasta:
    def test_export_fasta(self, manager: LibraryManager) -> None:
        fasta_path = f"test_export_{manager.session_id}.fasta"
        _CLEANUP_DIRS.append(fasta_path)  # won't be a dir but we handle below
        df = pd.DataFrame(
            {
                "variant_id": ["V1", "V2"],
                "aa_sequence": ["ACDEF", "GHIKL"],
            }
        )
        manager.export_fasta(df, fasta_path)
        with open(fasta_path) as fh:
            content = fh.read()
        assert ">V1" in content
        assert "ACDEF" in content
        # clean up file (not dir)
        if os.path.isfile(fasta_path):
            os.remove(fasta_path)
