"""Tests for vhh_library.tags – TagManager class."""

from __future__ import annotations

import pytest

from vhh_library.tags import TagManager

SAMPLE_VHH = (
    "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAISW"
    "SGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAAGVRAEWDYWGQGTLVTVSS"
)


@pytest.fixture
def tag_mgr() -> TagManager:
    return TagManager()


class TestLoadTags:
    def test_loads_tags(self, tag_mgr: TagManager) -> None:
        tags = tag_mgr.get_available_tags()
        assert isinstance(tags, dict)
        assert "6xHis" in tags


class TestBuildConstruct:
    def test_build_construct_with_c_tag(self, tag_mgr: TagManager) -> None:
        result = tag_mgr.build_construct(
            aa_sequence=SAMPLE_VHH,
            dna_sequence="ATG" * len(SAMPLE_VHH),
            c_tag="6xHis",
        )
        assert "HHHHHH" in result["aa_construct"]

    def test_build_construct_schematic(self, tag_mgr: TagManager) -> None:
        result = tag_mgr.build_construct(
            aa_sequence=SAMPLE_VHH,
            dna_sequence="ATG" * len(SAMPLE_VHH),
            c_tag="6xHis",
        )
        assert "VHH" in result["schematic"]
