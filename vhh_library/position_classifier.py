"""Rule-based IMGT position classifier for VHH library design.

Classifies each IMGT-numbered position as **frozen**, **conservative**, or
**mutable** based on a layered set of rules:

1. CDR positions are frozen by default.
2. Conserved structural positions (Cys-23, Trp-41, Cys-104) are frozen.
3. Selected framework support / core positions are marked conservative
   with restricted allowed amino-acid sets.
4. Remaining framework positions are mutable.
5. User overrides (from YAML or JSON) take highest precedence.

Every classification carries a :class:`ClassificationReason` that records
*why* a position received its class, enabling transparency and debugging.

The classifier produces a :class:`~vhh_library.position_policy.DesignPolicy`
as its output, so it integrates seamlessly with the existing mutation engine.

No hard dependency on structure models — all rules in v1 are sequence-based.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from vhh_library.position_policy import (
    DesignPolicy,
    PositionClass,
    PositionPolicy,
    imgt_region_for,
    parse_imgt_position,
)
from vhh_library.utils import AMINO_ACIDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conserved structural positions (always frozen by default)
# ---------------------------------------------------------------------------

_CONSERVED_POSITIONS: dict[str, str] = {
    "23": "C",  # disulfide bond
    "41": "W",  # conserved tryptophan
    "104": "C",  # disulfide bond
}

# ---------------------------------------------------------------------------
# Framework support / core positions — conservative by default
# ---------------------------------------------------------------------------
# These IMGT positions play structural roles in the VHH β-sandwich core or
# at the VH/VL interface (repurposed in camelid VHHs).  Restricting them to
# chemically similar residues reduces the risk of destabilisation.
#
# The allowed-residue sets are intentionally permissive in v1; future
# versions may tighten them using structural or statistical data.

_CONSERVATIVE_POSITIONS: dict[str, frozenset[str]] = {
    # FR1 core packing
    "6": frozenset({"A", "G", "S", "T", "V"}),
    "7": frozenset({"S", "T", "A", "G"}),
    # FR2 — hallmark VHH positions (42, 49) and structural neighbours
    "42": frozenset({"F", "Y", "K", "R", "E", "Q"}),
    "49": frozenset({"A", "G", "S", "E", "Q"}),
    # FR3 core / hydrophobic packing
    "69": frozenset({"I", "L", "V", "M", "F"}),
    "78": frozenset({"A", "V", "L", "I", "F"}),
    "80": frozenset({"L", "M", "I", "V"}),
    # FR4 — conserved structural residues
    "118": frozenset({"W", "F", "Y", "L"}),
}


# ---------------------------------------------------------------------------
# ClassificationReason
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ClassificationReason:
    """Metadata explaining *why* a position received its classification.

    Parameters
    ----------
    rule : str
        Short machine-readable rule tag (e.g. ``"cdr_freeze"``,
        ``"conserved_residue"``, ``"user_override"``).
    description : str
        Human-readable explanation.
    source : str
        Where the rule originated (``"default"``, ``"builtin"``, or a
        file path for user overrides).
    """

    rule: str
    description: str
    source: str = "builtin"


# ---------------------------------------------------------------------------
# PositionClassification
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PositionClassification:
    """Complete classification for one IMGT position.

    Parameters
    ----------
    imgt_position : str
        Canonical IMGT position string.
    position_class : PositionClass
        Assigned mutability class.
    allowed_aas : frozenset[str] | None
        Restricted residue set (only for CONSERVATIVE).
    reason : ClassificationReason
        Why this classification was assigned.
    """

    imgt_position: str
    position_class: PositionClass
    allowed_aas: frozenset[str] | None = None
    reason: ClassificationReason = dataclasses.field(
        default_factory=lambda: ClassificationReason("default", "No specific rule applied", "default"),
    )


# ---------------------------------------------------------------------------
# Override loader
# ---------------------------------------------------------------------------

# Override format (YAML or JSON):
#
# overrides:
#   - position: "23"
#     class: "frozen"                       # frozen | conservative | mutable
#     allowed_aas: null                     # only for conservative
#     reason: "User-specified freeze"       # optional human note
#
#   - position: "42"
#     class: "conservative"
#     allowed_aas: ["F", "Y"]
#     reason: "Restrict hallmark position"

_VALID_CLASSES = {"frozen", "conservative", "mutable"}


def _parse_override_entry(entry: dict[str, Any], source: str) -> tuple[str, PositionClass, frozenset[str] | None, str]:
    """Parse and validate one override entry.

    Returns (imgt_position, position_class, allowed_aas, reason_text).
    """
    raw_pos = entry.get("position")
    if raw_pos is None:
        raise ValueError(f"Override entry missing 'position' key: {entry}")
    pos_key = parse_imgt_position(raw_pos)

    raw_cls = entry.get("class")
    if raw_cls is None:
        raise ValueError(f"Override entry for position {pos_key} missing 'class' key")
    raw_cls_lower = str(raw_cls).lower().strip()
    if raw_cls_lower not in _VALID_CLASSES:
        raise ValueError(
            f"Override entry for position {pos_key}: invalid class {raw_cls!r}. Must be one of {sorted(_VALID_CLASSES)}"
        )
    pos_class = PositionClass(raw_cls_lower)

    raw_aas = entry.get("allowed_aas")
    allowed_aas: frozenset[str] | None = None
    if raw_aas is not None:
        if not isinstance(raw_aas, list):
            raise ValueError(
                f"Override entry for position {pos_key}: 'allowed_aas' must be a list, got {type(raw_aas).__name__}"
            )
        allowed_aas = frozenset(str(aa).upper() for aa in raw_aas)
        unknown = allowed_aas - AMINO_ACIDS
        if unknown:
            raise ValueError(f"Override entry for position {pos_key}: unknown amino acid(s) {sorted(unknown)}")

    reason_text = str(entry.get("reason", "User override"))

    # Consistency checks
    if pos_class is PositionClass.FROZEN and allowed_aas is not None:
        raise ValueError(f"Override for position {pos_key}: FROZEN must not specify allowed_aas")
    if pos_class is PositionClass.CONSERVATIVE and (allowed_aas is None or len(allowed_aas) == 0):
        raise ValueError(f"Override for position {pos_key}: CONSERVATIVE must specify non-empty allowed_aas")
    if pos_class is PositionClass.MUTABLE and allowed_aas is not None:
        raise ValueError(f"Override for position {pos_key}: MUTABLE must not specify allowed_aas")

    return pos_key, pos_class, allowed_aas, reason_text


def load_overrides(path: str | Path) -> list[dict[str, Any]]:
    """Load position overrides from a YAML or JSON file.

    The file must contain a top-level ``overrides`` key with a list of
    override entries.  See module docstring for the expected format.

    Parameters
    ----------
    path : str | Path
        Path to a ``.yaml``, ``.yml``, or ``.json`` override file.

    Returns
    -------
    list[dict]
        Raw override entries (not yet validated).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported or the structure is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Override file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML override files. Install it with: pip install pyyaml"
            ) from exc
        with open(p) as fh:
            data = yaml.safe_load(fh)
    elif suffix == ".json":
        with open(p) as fh:
            data = json.load(fh)
    else:
        raise ValueError(f"Unsupported override file format: {suffix!r}. Use .yaml, .yml, or .json")

    if not isinstance(data, dict) or "overrides" not in data:
        raise ValueError(f"Override file must contain a top-level 'overrides' key: {p}")

    overrides = data["overrides"]
    if not isinstance(overrides, list):
        raise ValueError(f"'overrides' must be a list: {p}")

    return overrides


# ---------------------------------------------------------------------------
# PositionClassifier
# ---------------------------------------------------------------------------


class PositionClassifier:
    """Rule-based classifier that assigns a mutability class to each IMGT position.

    The classifier applies rules in the following precedence order
    (highest → lowest):

    1. **User overrides** — loaded from YAML/JSON or passed as dicts.
    2. **Conserved-residue freeze** — positions 23 (Cys), 41 (Trp), 104 (Cys).
    3. **CDR freeze** — all positions inside CDR1/CDR2/CDR3 regions.
    4. **Conservative framework positions** — selected core/support positions
       with restricted allowed amino-acid sets.
    5. **Default mutable** — remaining framework positions.

    Parameters
    ----------
    freeze_cdrs : bool
        When ``True`` (default), CDR positions are frozen.
    freeze_conserved : bool
        When ``True`` (default), conserved structural positions are frozen.
    conservative_positions : dict[str, frozenset[str]] | None
        Custom conservative-position map.  If ``None``, the built-in
        ``_CONSERVATIVE_POSITIONS`` is used.
    overrides : list[dict] | None
        Override entries (same format as the file contents).
    override_file : str | Path | None
        Path to a YAML/JSON override file.  Loaded overrides are
        appended **after** any entries in *overrides*, so file entries
        take precedence (last writer wins for duplicate positions).
    """

    def __init__(
        self,
        *,
        freeze_cdrs: bool = True,
        freeze_conserved: bool = True,
        conservative_positions: dict[str, frozenset[str]] | None = None,
        overrides: list[dict[str, Any]] | None = None,
        override_file: str | Path | None = None,
    ) -> None:
        self.freeze_cdrs = freeze_cdrs
        self.freeze_conserved = freeze_conserved
        self.conservative_positions = (
            dict(conservative_positions) if conservative_positions is not None else dict(_CONSERVATIVE_POSITIONS)
        )

        # Merge overrides: inline first, then file (file entries win).
        # Each entry is a (dict, source_label) pair so that reason metadata
        # correctly attributes each override to inline or file origin.
        self._override_entries: list[tuple[dict[str, Any], str]] = [(entry, "inline") for entry in (overrides or [])]
        if override_file is not None:
            file_source = str(override_file)
            file_overrides = load_overrides(override_file)
            self._override_entries.extend((entry, file_source) for entry in file_overrides)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        imgt_positions: list[str],
    ) -> dict[str, PositionClassification]:
        """Classify every position in *imgt_positions*.

        Parameters
        ----------
        imgt_positions : list[str]
            IMGT position keys present in the sequence (e.g. from
            ``VHHSequence.imgt_numbered.keys()``).

        Returns
        -------
        dict[str, PositionClassification]
            Map from IMGT position string to its classification.
        """
        result: dict[str, PositionClassification] = {}

        for raw_pos in imgt_positions:
            pos_key = parse_imgt_position(raw_pos)
            result[pos_key] = self._classify_single(pos_key)

        # Apply user overrides last (highest precedence).
        self._apply_overrides(result)

        return result

    def classify_vhh(self, vhh: object) -> dict[str, PositionClassification]:
        """Convenience: classify all positions of a VHHSequence.

        Parameters
        ----------
        vhh : VHHSequence
            A numbered VHH sequence.

        Returns
        -------
        dict[str, PositionClassification]

        Raises
        ------
        TypeError
            If *vhh* does not have an ``imgt_numbered`` attribute.
        """
        if not hasattr(vhh, "imgt_numbered"):
            raise TypeError(f"Expected a VHHSequence-like object, got {type(vhh).__name__}")
        return self.classify(list(vhh.imgt_numbered.keys()))  # type: ignore[union-attr]

    def to_design_policy(
        self,
        imgt_positions: list[str],
    ) -> DesignPolicy:
        """Classify positions and return a :class:`DesignPolicy`.

        This is the primary integration point with the mutation engine.
        """
        classifications = self.classify(imgt_positions)
        return _classifications_to_policy(classifications)

    # ------------------------------------------------------------------
    # Internal rule engine
    # ------------------------------------------------------------------

    def _classify_single(self, pos_key: str) -> PositionClassification:
        """Apply built-in rules to a single position (before overrides)."""
        region = imgt_region_for(pos_key)

        # Rule 1: Conserved structural positions (always frozen).
        if self.freeze_conserved and pos_key in _CONSERVED_POSITIONS:
            expected_aa = _CONSERVED_POSITIONS[pos_key]
            return PositionClassification(
                imgt_position=pos_key,
                position_class=PositionClass.FROZEN,
                reason=ClassificationReason(
                    rule="conserved_residue",
                    description=f"Conserved {expected_aa} at IMGT {pos_key} (structural requirement)",
                ),
            )

        # Rule 2: CDR freeze.
        if self.freeze_cdrs and region is not None and region.startswith("CDR"):
            return PositionClassification(
                imgt_position=pos_key,
                position_class=PositionClass.FROZEN,
                reason=ClassificationReason(
                    rule="cdr_freeze",
                    description=f"CDR position ({region}) frozen by default",
                ),
            )

        # Rule 3: Conservative framework positions.
        if pos_key in self.conservative_positions:
            allowed = self.conservative_positions[pos_key]
            return PositionClassification(
                imgt_position=pos_key,
                position_class=PositionClass.CONSERVATIVE,
                allowed_aas=allowed,
                reason=ClassificationReason(
                    rule="framework_support",
                    description=f"Framework support/core position at IMGT {pos_key}",
                ),
            )

        # Rule 4: Default mutable.
        return PositionClassification(
            imgt_position=pos_key,
            position_class=PositionClass.MUTABLE,
            reason=ClassificationReason(
                rule="default_mutable",
                description=f"Framework position IMGT {pos_key} — eligible for mutation",
                source="default",
            ),
        )

    def _apply_overrides(self, classifications: dict[str, PositionClassification]) -> None:
        """Apply user overrides, mutating *classifications* in-place."""
        for entry, source in self._override_entries:
            pos_key, pos_class, allowed_aas, reason_text = _parse_override_entry(entry, source)
            classifications[pos_key] = PositionClassification(
                imgt_position=pos_key,
                position_class=pos_class,
                allowed_aas=allowed_aas,
                reason=ClassificationReason(
                    rule="user_override",
                    description=reason_text,
                    source=source,
                ),
            )


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _classifications_to_policy(
    classifications: dict[str, PositionClassification],
) -> DesignPolicy:
    """Convert a classification map to a :class:`DesignPolicy`."""
    dp = DesignPolicy()
    for pos_key, clf in classifications.items():
        dp.policies[pos_key] = PositionPolicy(
            imgt_position=pos_key,
            position_class=clf.position_class,
            allowed_aas=clf.allowed_aas,
        )
    return dp
