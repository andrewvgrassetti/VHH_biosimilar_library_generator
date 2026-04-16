"""Cross-platform wrapper for ``abnativ init``.

This thin entry-point applies :func:`vhh_library._abnativ_compat.patch_abnativ_platform`
before delegating to the upstream ``abnativ`` CLI, so that model-weight
downloads work on Windows (where ``os.uname`` is unavailable).

Usage (after ``pip install -e .``)::

    vhh-init          # equivalent to: abnativ init
    vhh-init --force  # equivalent to: abnativ init --force
"""

from __future__ import annotations

import sys


def main() -> None:
    """Patch abnativ platform detection and delegate to ``abnativ`` CLI."""
    from vhh_library._abnativ_compat import patch_abnativ_platform

    patch_abnativ_platform()

    from abnativ.__main__ import main as _abnativ_main

    # abnativ's main() reads sys.argv; rewrite argv[0] so help text
    # still looks correct, then forward all arguments.
    sys.argv[0] = "abnativ"
    _abnativ_main()


if __name__ == "__main__":
    main()
