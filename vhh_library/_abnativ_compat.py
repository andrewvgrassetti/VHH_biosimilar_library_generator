"""Cross-platform compatibility shim for the AbNatiV package.

AbNatiV's ``init`` module calls ``os.uname().sysname`` at import time to
determine the model-weights directory.  ``os.uname`` is a Unix-only API and
raises ``AttributeError`` on Windows, making both ``abnativ init`` and any
code path that transitively imports ``abnativ.init`` (including the scoring
functions) crash before they can do useful work.

This module provides :func:`patch_abnativ_platform` which **must** be called
before ``abnativ.init`` is first imported.  It monkey-patches
``abnativ.init.get_platform`` (and the derived ``PRETRAINED_MODELS_DIR``
constant) so that the standard-library :func:`platform.system` is used
instead of ``os.uname``.
"""

from __future__ import annotations

import os
import platform

_PATCHED = False


def patch_abnativ_platform() -> None:
    """Ensure ``abnativ.init`` can be imported on every platform.

    On Windows (or any system where ``os.uname`` is unavailable) this
    installs a thin ``os.uname`` shim *before* ``abnativ.init`` is loaded
    so that its module-level ``get_platform()`` call succeeds.

    The function is idempotent — calling it more than once is harmless.
    """
    global _PATCHED  # noqa: PLW0603
    if _PATCHED:
        return

    if not hasattr(os, "uname"):
        # Create a minimal os.uname() stand-in that returns a named-tuple-like
        # object with a ``sysname`` attribute, which is the only field
        # abnativ.init.get_platform() reads.
        _sysname = platform.system()  # "Windows", "Linux", "Darwin", …

        class _UnameResult:
            """Minimal ``os.uname()`` replacement for Windows."""

            sysname = _sysname
            nodename = platform.node()
            release = platform.release()
            version = platform.version()
            machine = platform.machine()

        os.uname = lambda: _UnameResult()  # type: ignore[attr-defined]

    _PATCHED = True
