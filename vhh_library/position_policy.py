"""Position-policy domain model for three-class mutation design.

This module introduces a structured way to classify each IMGT-numbered
position of a VHH sequence into one of three mutability classes:

* **FROZEN** — the position must not be mutated under any circumstances.
* **CONSERVATIVE** — only a restricted set of amino acids is allowed
  (e.g. chemically similar residues).
* **MUTABLE** — any standard amino acid substitution may be proposed
  (subject to downstream PTM / liability filters).

The default classification is inferred automatically from IMGT region
annotations (CDR positions → FROZEN, conserved cysteines/Trp → FROZEN,
framework positions → MUTABLE), but every position can be overridden by
the user at construction time or afterwards.

Legacy adapter helpers convert the existing ``off_limits`` /
``forbidden_substitutions`` controls used by :class:`MutationEngine` into
a :class:`DesignPolicy` and back, so that **no behavioural change** is
introduced by this PR.

All position identifiers are **IMGT position strings** (``"1"``, ``"27"``,
``"111A"``, …).  Integer inputs are accepted for backward compatibility
and immediately coerced to ``str``.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Iterable

from vhh_library.sequence import IMGT_REGIONS
from vhh_library.utils import AMINO_ACIDS

# ---------------------------------------------------------------------------
# IMGT position helpers
# ---------------------------------------------------------------------------

# Regex that matches a valid IMGT position key: one or more digits,
# optionally followed by a single uppercase letter (insertion code).
_IMGT_POS_RE = re.compile(r"^(\d+)([A-Z]?)$")

# Conserved IMGT positions that should be frozen by default.
_CONSERVED_POSITIONS: dict[str, str] = {
    "23": "C",
    "41": "W",
    "104": "C",
}


def parse_imgt_position(pos: int | str) -> str:
    """Normalise *pos* to a canonical IMGT position string.

    Accepts plain integers (``1``, ``111``), integer-like strings
    (``"1"``, ``"111"``), and insertion-coded strings (``"111A"``).

    Parameters
    ----------
    pos : int | str
        An IMGT position identifier.

    Returns
    -------
    str
        Canonical string key (e.g. ``"1"``, ``"111A"``).

    Raises
    ------
    ValueError
        If *pos* cannot be parsed as a valid IMGT position.
    """
    if isinstance(pos, int):
        if pos < 1:
            raise ValueError(f"IMGT position must be >= 1, got {pos}")
        return str(pos)

    pos_str = str(pos).strip()
    if not _IMGT_POS_RE.match(pos_str):
        raise ValueError(f"Invalid IMGT position string: {pos_str!r}")
    # Strip leading zeros for consistency (e.g. "01" → "1").
    m = _IMGT_POS_RE.match(pos_str)
    assert m is not None  # guaranteed by check above
    num, code = m.groups()
    return f"{int(num)}{code}"


def imgt_base_number(pos_key: str) -> int:
    """Extract the integer portion from an IMGT position key.

    ``"111A"`` → ``111``, ``"27"`` → ``27``.

    Parameters
    ----------
    pos_key : str
        Canonical IMGT position string.

    Returns
    -------
    int
        The numeric (non-insertion) portion.
    """
    digits = "".join(c for c in pos_key if c.isdigit())
    if not digits:
        raise ValueError(f"No numeric portion found in IMGT position key: {pos_key!r}")
    return int(digits)


def imgt_region_for(pos_key: str) -> str | None:
    """Return the IMGT region name (e.g. ``"CDR1"``, ``"FR3"``) for *pos_key*.

    For insertion positions (e.g. ``"111A"``), the base number is used to
    determine the region.

    Returns ``None`` if the position falls outside all defined regions.
    """
    base = imgt_base_number(pos_key)
    for name, (start, end) in IMGT_REGIONS.items():
        if start <= base <= end:
            return name
    return None


# ---------------------------------------------------------------------------
# PositionClass enum
# ---------------------------------------------------------------------------


class PositionClass(enum.Enum):
    """Mutability class for a single IMGT position.

    Members
    -------
    FROZEN
        Position must not be mutated.
    CONSERVATIVE
        Only specific (user-defined) amino acids are allowed.
    MUTABLE
        Any standard amino acid substitution may be proposed.
    """

    FROZEN = "frozen"
    CONSERVATIVE = "conservative"
    MUTABLE = "mutable"


# ---------------------------------------------------------------------------
# PositionPolicy dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PositionPolicy:
    """Mutation policy for a single IMGT position.

    Parameters
    ----------
    imgt_position : str
        Canonical IMGT position string (e.g. ``"27"``, ``"111A"``).
    position_class : PositionClass
        Mutability class.
    allowed_aas : frozenset[str] | None
        For :attr:`PositionClass.CONSERVATIVE`, the set of amino acids
        that *may* replace the wild-type residue.  Must be ``None`` for
        FROZEN and MUTABLE positions.
    """

    imgt_position: str
    position_class: PositionClass
    allowed_aas: frozenset[str] | None = None

    def __post_init__(self) -> None:
        # Validate imgt_position format.
        if not _IMGT_POS_RE.match(self.imgt_position):
            raise ValueError(f"Invalid IMGT position string: {self.imgt_position!r}")

        if self.position_class is PositionClass.FROZEN and self.allowed_aas is not None:
            raise ValueError("FROZEN positions must not specify allowed_aas")

        if self.position_class is PositionClass.CONSERVATIVE:
            if self.allowed_aas is None or len(self.allowed_aas) == 0:
                raise ValueError("CONSERVATIVE positions must specify at least one allowed_aas")
            unknown = self.allowed_aas - AMINO_ACIDS
            if unknown:
                raise ValueError(
                    f"Unknown amino acid(s) in allowed_aas: {sorted(unknown)}. "
                    f"Valid amino acids are: {sorted(AMINO_ACIDS)}"
                )

        if self.position_class is PositionClass.MUTABLE and self.allowed_aas is not None:
            raise ValueError("MUTABLE positions must not specify allowed_aas (all AAs are allowed)")

    # -- Convenience predicates -------------------------------------------

    @property
    def is_frozen(self) -> bool:
        """Return ``True`` if the position is frozen."""
        return self.position_class is PositionClass.FROZEN

    @property
    def is_conservative(self) -> bool:
        """Return ``True`` if the position is conservatively mutable."""
        return self.position_class is PositionClass.CONSERVATIVE

    @property
    def is_mutable(self) -> bool:
        """Return ``True`` if the position is fully mutable."""
        return self.position_class is PositionClass.MUTABLE

    def permits(self, aa: str) -> bool:
        """Return ``True`` if substituting *aa* is permitted by this policy.

        FROZEN → always ``False``.
        CONSERVATIVE → ``True`` iff *aa* is in :attr:`allowed_aas`.
        MUTABLE → ``True`` for any valid amino acid.
        """
        if self.position_class is PositionClass.FROZEN:
            return False
        if self.position_class is PositionClass.CONSERVATIVE:
            assert self.allowed_aas is not None
            return aa in self.allowed_aas
        return aa in AMINO_ACIDS

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        d: dict = {
            "imgt_position": self.imgt_position,
            "position_class": self.position_class.value,
        }
        if self.allowed_aas is not None:
            d["allowed_aas"] = sorted(self.allowed_aas)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PositionPolicy:
        """Deserialise from a plain dictionary."""
        allowed = data.get("allowed_aas")
        return cls(
            imgt_position=data["imgt_position"],
            position_class=PositionClass(data["position_class"]),
            allowed_aas=frozenset(allowed) if allowed is not None else None,
        )


# ---------------------------------------------------------------------------
# DesignPolicy dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DesignPolicy:
    """Collection of :class:`PositionPolicy` entries keyed by IMGT position.

    A ``DesignPolicy`` describes the mutation rules for an entire VHH
    sequence.  Positions not explicitly listed default to **MUTABLE** in
    framework regions and **FROZEN** in CDR regions (matching the current
    mutation-engine behaviour).

    Parameters
    ----------
    policies : dict[str, PositionPolicy]
        Map from IMGT position string to its policy.
    """

    policies: dict[str, PositionPolicy] = dataclasses.field(default_factory=dict)

    # -- Query helpers ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.policies)

    def __contains__(self, pos: int | str) -> bool:
        return parse_imgt_position(pos) in self.policies

    def __getitem__(self, pos: int | str) -> PositionPolicy:
        return self.policies[parse_imgt_position(pos)]

    def get(self, pos: int | str) -> PositionPolicy | None:
        """Return the policy for *pos*, or ``None`` if not explicitly set."""
        return self.policies.get(parse_imgt_position(pos))

    def effective_class(self, pos: int | str) -> PositionClass:
        """Return the effective class for *pos*.

        If the position has an explicit policy, return its class.
        Otherwise, infer based on IMGT region: CDR → FROZEN,
        framework → MUTABLE.
        """
        key = parse_imgt_position(pos)
        if key in self.policies:
            return self.policies[key].position_class
        region = imgt_region_for(key)
        if region is not None and region.startswith("CDR"):
            return PositionClass.FROZEN
        return PositionClass.MUTABLE

    def permits(self, pos: int | str, aa: str) -> bool:
        """Return ``True`` if substituting *aa* at *pos* is permitted.

        Delegates to the explicit :class:`PositionPolicy` if one exists,
        otherwise falls back to the region-based default (CDR → frozen,
        framework → mutable).
        """
        key = parse_imgt_position(pos)
        policy = self.policies.get(key)
        if policy is not None:
            return policy.permits(aa)
        # Default: CDR frozen, framework mutable.
        region = imgt_region_for(key)
        if region is not None and region.startswith("CDR"):
            return False
        return aa in AMINO_ACIDS

    # -- Bulk queries -----------------------------------------------------

    def frozen_positions(self) -> frozenset[str]:
        """Return all explicitly FROZEN positions."""
        return frozenset(k for k, v in self.policies.items() if v.position_class is PositionClass.FROZEN)

    def conservative_positions(self) -> frozenset[str]:
        """Return all explicitly CONSERVATIVE positions."""
        return frozenset(k for k, v in self.policies.items() if v.position_class is PositionClass.CONSERVATIVE)

    def mutable_positions(self) -> frozenset[str]:
        """Return all explicitly MUTABLE positions."""
        return frozenset(k for k, v in self.policies.items() if v.position_class is PositionClass.MUTABLE)

    # -- Mutation helpers -------------------------------------------------

    def set_policy(self, policy: PositionPolicy) -> None:
        """Add or replace the policy for a single position."""
        self.policies[policy.imgt_position] = policy

    def freeze(self, positions: Iterable[int | str]) -> None:
        """Mark *positions* as FROZEN."""
        for pos in positions:
            key = parse_imgt_position(pos)
            self.policies[key] = PositionPolicy(key, PositionClass.FROZEN)

    def make_mutable(self, positions: Iterable[int | str]) -> None:
        """Mark *positions* as MUTABLE (removing any restriction)."""
        for pos in positions:
            key = parse_imgt_position(pos)
            self.policies[key] = PositionPolicy(key, PositionClass.MUTABLE)

    def restrict(self, pos: int | str, allowed_aas: Iterable[str]) -> None:
        """Mark *pos* as CONSERVATIVE with the given allowed amino acids."""
        key = parse_imgt_position(pos)
        self.policies[key] = PositionPolicy(key, PositionClass.CONSERVATIVE, frozenset(allowed_aas))

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {"policies": {k: v.to_dict() for k, v in self.policies.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> DesignPolicy:
        """Deserialise from a plain dictionary."""
        policies = {k: PositionPolicy.from_dict(v) for k, v in data.get("policies", {}).items()}
        return cls(policies=policies)

    # -- Repr -------------------------------------------------------------

    def __repr__(self) -> str:
        counts = {
            "frozen": sum(1 for v in self.policies.values() if v.is_frozen),
            "conservative": sum(1 for v in self.policies.values() if v.is_conservative),
            "mutable": sum(1 for v in self.policies.values() if v.is_mutable),
        }
        return (
            f"DesignPolicy(frozen={counts['frozen']}, "
            f"conservative={counts['conservative']}, "
            f"mutable={counts['mutable']})"
        )


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def default_design_policy(
    imgt_positions: Iterable[str] | None = None,
    *,
    freeze_cdrs: bool = True,
    freeze_conserved: bool = True,
) -> DesignPolicy:
    """Build a default :class:`DesignPolicy` from a set of IMGT positions.

    Parameters
    ----------
    imgt_positions : iterable of str, optional
        The IMGT position keys present in the sequence (e.g. from
        ``VHHSequence.imgt_numbered.keys()``).  If ``None``, only the
        conserved-residue policies are created.
    freeze_cdrs : bool
        When ``True`` (default), all CDR positions are marked FROZEN.
    freeze_conserved : bool
        When ``True`` (default), conserved IMGT positions (23, 41, 104)
        are marked FROZEN.

    Returns
    -------
    DesignPolicy
    """
    dp = DesignPolicy()

    if imgt_positions is not None:
        for pos_key in imgt_positions:
            region = imgt_region_for(pos_key)
            if freeze_cdrs and region is not None and region.startswith("CDR"):
                dp.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)
            else:
                dp.policies[pos_key] = PositionPolicy(pos_key, PositionClass.MUTABLE)

    if freeze_conserved:
        for pos_key in _CONSERVED_POSITIONS:
            dp.policies[pos_key] = PositionPolicy(pos_key, PositionClass.FROZEN)

    return dp


# ---------------------------------------------------------------------------
# Legacy adapters
# ---------------------------------------------------------------------------


def from_off_limits(
    off_limits: set[int] | set[str] | None = None,
    forbidden_substitutions: dict[int, set[str]] | dict[str, set[str]] | None = None,
    *,
    imgt_positions: Iterable[str] | None = None,
    freeze_cdrs: bool = True,
) -> DesignPolicy:
    """Convert legacy mutation-engine controls into a :class:`DesignPolicy`.

    This adapter produces a ``DesignPolicy`` that is **semantically
    equivalent** to the existing ``off_limits`` + ``forbidden_substitutions``
    parameters of :meth:`MutationEngine.rank_single_mutations`.

    Parameters
    ----------
    off_limits : set of int or str, optional
        IMGT positions that must not be mutated (→ FROZEN).
    forbidden_substitutions : dict, optional
        Map from IMGT position to a set of amino acids that are **not**
        allowed.  Positions in this dict that are *not* in ``off_limits``
        become CONSERVATIVE (with ``allowed_aas = AMINO_ACIDS - forbidden``).
    imgt_positions : iterable of str, optional
        All IMGT position keys in the sequence.  When provided, CDR
        positions not already in ``off_limits`` are also frozen.
    freeze_cdrs : bool
        When ``True`` (default), CDR positions found in *imgt_positions*
        are frozen.  This matches the current engine behaviour.

    Returns
    -------
    DesignPolicy
    """
    dp = default_design_policy(imgt_positions, freeze_cdrs=freeze_cdrs, freeze_conserved=False)

    if off_limits:
        for pos in off_limits:
            key = parse_imgt_position(pos)
            dp.policies[key] = PositionPolicy(key, PositionClass.FROZEN)

    if forbidden_substitutions:
        for pos, forbidden_set in forbidden_substitutions.items():
            key = parse_imgt_position(pos)
            # If the position is already frozen, keep it frozen.
            existing = dp.policies.get(key)
            if existing is not None and existing.is_frozen:
                continue
            allowed = AMINO_ACIDS - frozenset(forbidden_set)
            if not allowed:
                # All AAs forbidden → effectively frozen.
                dp.policies[key] = PositionPolicy(key, PositionClass.FROZEN)
            else:
                dp.policies[key] = PositionPolicy(key, PositionClass.CONSERVATIVE, frozenset(allowed))

    return dp


def to_off_limits(
    policy: DesignPolicy,
) -> tuple[set[str], dict[str, set[str]]]:
    """Convert a :class:`DesignPolicy` back to legacy mutation-engine controls.

    Returns
    -------
    off_limits : set[str]
        IMGT position strings that should be treated as off-limits.
    forbidden_substitutions : dict[str, set[str]]
        Map from IMGT position string to the set of forbidden amino acids.
    """
    off_limits: set[str] = set()
    forbidden_substitutions: dict[str, set[str]] = {}

    for pos_key, pp in policy.policies.items():
        if pp.is_frozen:
            off_limits.add(pos_key)
        elif pp.is_conservative:
            assert pp.allowed_aas is not None
            forbidden = AMINO_ACIDS - pp.allowed_aas
            if forbidden:
                forbidden_substitutions[pos_key] = set(forbidden)

    return off_limits, forbidden_substitutions


def from_vhh_sequence(
    vhh: object,
    *,
    freeze_cdrs: bool = True,
    freeze_conserved: bool = True,
) -> DesignPolicy:
    """Build a :class:`DesignPolicy` from a :class:`VHHSequence` instance.

    This is a convenience wrapper around :func:`default_design_policy`
    that reads the IMGT positions directly from the sequence object.

    Parameters
    ----------
    vhh : VHHSequence
        A numbered VHH sequence.  The type hint is ``object`` to avoid
        a circular import; at runtime it must be a ``VHHSequence``.
    freeze_cdrs : bool
        Freeze CDR positions (default ``True``).
    freeze_conserved : bool
        Freeze conserved positions (default ``True``).

    Returns
    -------
    DesignPolicy

    Raises
    ------
    TypeError
        If *vhh* does not have an ``imgt_numbered`` attribute.
    """
    if not hasattr(vhh, "imgt_numbered"):
        raise TypeError(f"Expected a VHHSequence-like object, got {type(vhh).__name__}")
    imgt_positions = list(vhh.imgt_numbered.keys())  # type: ignore[union-attr]
    return default_design_policy(
        imgt_positions,
        freeze_cdrs=freeze_cdrs,
        freeze_conserved=freeze_conserved,
    )
