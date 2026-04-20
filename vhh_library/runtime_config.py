"""Centralized runtime / backend configuration for model-backed scorers.

This module provides :class:`RuntimeConfig`, a frozen dataclass that
captures device selection, backend preferences, batch-size overrides,
cache directory, and logging knobs in a single place.

It also exposes :func:`resolve_device`, the shared device-resolution
helper referenced by the predictor instruction set
(``vhh_library.device_utils.resolve_device`` — aliased here until a
dedicated ``device_utils`` module is warranted).

**Default stability backend is now NanoMelt.**  ``RuntimeConfig()`` yields
NanoMelt stability on auto-detected device with AbNatiV nativeness.

Environment-variable construction (``RuntimeConfig.from_env()``) enables
headless / AWS workstation use without touching Python code.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import warnings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid option sets
# ---------------------------------------------------------------------------

VALID_DEVICES: frozenset[str] = frozenset({"auto", "cpu", "cuda"})
VALID_STABILITY_BACKENDS: frozenset[str] = frozenset({"esm2", "nanomelt", "both"})
VALID_NATIVENESS_BACKENDS: frozenset[str] = frozenset({"abnativ"})

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes"})


# ---------------------------------------------------------------------------
# Device resolution (shared helper)
# ---------------------------------------------------------------------------


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string to a concrete PyTorch device name.

    Resolution order for ``"auto"``:

    1. ``"cuda"`` if ``torch.cuda.is_available()``
    2. ``"mps"``  if ``torch.backends.mps.is_available()`` (Apple Silicon)
    3. ``"cpu"``

    If the requested device is unavailable, falls back to ``"cpu"`` and
    emits a :class:`RuntimeWarning`.

    Parameters
    ----------
    device : str
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns
    -------
    str
        Concrete device name suitable for ``torch.device()``.
    """
    if device == "auto":
        try:
            import torch  # noqa: WPS433

            if torch.cuda.is_available():
                try:
                    torch.zeros(1, device="cuda")
                    return "cuda"
                except Exception:
                    warnings.warn(
                        "CUDA reported as available but failed a smoke test; falling back to CPU.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    if device == "cuda":
        try:
            import torch  # noqa: WPS433

            if not torch.cuda.is_available():
                warnings.warn(
                    "CUDA requested but not available; falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return "cpu"
            try:
                torch.zeros(1, device="cuda")
            except Exception:
                warnings.warn(
                    "CUDA reported as available but failed a smoke test; falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return "cpu"
        except ImportError:
            warnings.warn(
                "CUDA requested but torch is not installed; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return "cuda"

    if device == "mps":
        try:
            import torch  # noqa: WPS433

            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                warnings.warn(
                    "MPS requested but not available; falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return "cpu"
        except ImportError:
            warnings.warn(
                "MPS requested but torch is not installed; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        return "mps"

    # Unknown device string — pass through (torch will raise later if invalid)
    return device


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Immutable runtime configuration for model-backed scorers.

    Parameters
    ----------
    device : str
        Device for PyTorch inference: ``"auto"``, ``"cpu"``, or ``"cuda"``.
    stability_backend : str
        Stability scoring backend: ``"esm2"``, ``"nanomelt"``, or ``"both"``.
    nativeness_backend : str
        Nativeness scoring backend: ``"abnativ"``.
    batch_size : int | None
        Global batch-size override.  ``None`` defers to per-backend defaults.
    cache_dir : str | None
        Directory for score caches (e.g. ESM-2 SQLite cache).
        ``None`` disables caching.
    verbose : bool
        When *True*, set the ``vhh_library`` logger to ``INFO``.
    debug : bool
        When *True*, set the ``vhh_library`` logger to ``DEBUG``
        (takes precedence over *verbose*).
    """

    device: str = "auto"
    stability_backend: str = "nanomelt"
    nativeness_backend: str = "abnativ"
    batch_size: int | None = None
    cache_dir: str | None = None
    verbose: bool = False
    debug: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.device not in VALID_DEVICES:
            raise ValueError(f"Invalid device {self.device!r}; choose from {sorted(VALID_DEVICES)}")
        if self.stability_backend not in VALID_STABILITY_BACKENDS:
            raise ValueError(
                f"Invalid stability_backend {self.stability_backend!r}; choose from {sorted(VALID_STABILITY_BACKENDS)}"
            )
        if self.nativeness_backend not in VALID_NATIVENESS_BACKENDS:
            raise ValueError(
                f"Invalid nativeness_backend {self.nativeness_backend!r}; "
                f"choose from {sorted(VALID_NATIVENESS_BACKENDS)}"
            )
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        """Build a config from ``VHH_*`` environment variables.

        Supported variables (all optional — omitted variables use defaults):

        ``VHH_DEVICE``
            ``"auto"``, ``"cpu"``, or ``"cuda"``.
        ``VHH_STABILITY_BACKEND``
            ``"esm2"``, ``"nanomelt"``, or ``"both"``.
        ``VHH_NATIVENESS_BACKEND``
            ``"abnativ"``.
        ``VHH_BATCH_SIZE``
            Positive integer.
        ``VHH_CACHE_DIR``
            Filesystem path.
        ``VHH_VERBOSE``
            ``"1"``, ``"true"``, or ``"yes"`` to enable.
        ``VHH_DEBUG``
            ``"1"``, ``"true"``, or ``"yes"`` to enable.
        """
        kwargs: dict = {}

        if (v := os.environ.get("VHH_DEVICE")) is not None:
            kwargs["device"] = v.strip().lower()
        if (v := os.environ.get("VHH_STABILITY_BACKEND")) is not None:
            kwargs["stability_backend"] = v.strip().lower()
        if (v := os.environ.get("VHH_NATIVENESS_BACKEND")) is not None:
            kwargs["nativeness_backend"] = v.strip().lower()
        if (v := os.environ.get("VHH_BATCH_SIZE")) is not None:
            try:
                kwargs["batch_size"] = int(v.strip())
            except ValueError:
                warnings.warn(
                    f"VHH_BATCH_SIZE={v!r} is not a valid integer; ignoring.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if (v := os.environ.get("VHH_CACHE_DIR")) is not None:
            kwargs["cache_dir"] = v.strip()
        if (v := os.environ.get("VHH_VERBOSE")) is not None:
            kwargs["verbose"] = v.strip().lower() in _TRUTHY
        if (v := os.environ.get("VHH_DEBUG")) is not None:
            kwargs["debug"] = v.strip().lower() in _TRUTHY

        return cls(**kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> RuntimeConfig:
        """Build a config from a plain dictionary, ignoring unknown keys."""
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def resolve_device(self) -> str:
        """Return a concrete device string (``"cpu"``, ``"cuda"``, ``"mps"``)."""
        return resolve_device(self.device)

    def configure_logging(self) -> None:
        """Set the ``vhh_library`` package logger level based on config."""
        pkg_logger = logging.getLogger("vhh_library")
        if self.debug:
            pkg_logger.setLevel(logging.DEBUG)
        elif self.verbose:
            pkg_logger.setLevel(logging.INFO)

    def needs_nanomelt(self) -> bool:
        """Return *True* when the chosen stability backend requires NanoMelt."""
        return self.stability_backend in {"nanomelt", "both"}

    def needs_esm2(self) -> bool:
        """Return *True* when the chosen stability backend requires ESM-2."""
        return self.stability_backend in {"esm2", "both"}
