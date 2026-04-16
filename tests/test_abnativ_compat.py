"""Tests for vhh_library._abnativ_compat – cross-platform abnativ patch."""

from __future__ import annotations

import os
from unittest import mock

import pytest


class TestPatchAbnativPlatform:
    """Verify the os.uname shim installs correctly and is idempotent."""

    def test_idempotent_on_unix(self) -> None:
        """On Unix (where os.uname exists), patching does nothing harmful."""
        from vhh_library._abnativ_compat import patch_abnativ_platform

        original_uname = os.uname
        patch_abnativ_platform()
        # os.uname should still be the original function
        assert os.uname is original_uname

    def test_shim_installed_when_uname_missing(self) -> None:
        """When os.uname is absent, the patch installs a shim."""
        import vhh_library._abnativ_compat as compat_mod

        # Reset patch state so we can re-test
        compat_mod._PATCHED = False
        original_uname = os.uname

        try:
            del os.uname  # type: ignore[attr-defined]
            compat_mod.patch_abnativ_platform()

            # The shim should now exist
            assert hasattr(os, "uname")
            result = os.uname()
            assert hasattr(result, "sysname")
            assert isinstance(result.sysname, str)
            assert len(result.sysname) > 0
        finally:
            # Restore original os.uname
            os.uname = original_uname  # type: ignore[attr-defined]
            compat_mod._PATCHED = False

    def test_shim_sysname_matches_platform(self) -> None:
        """The shim's sysname should match platform.system()."""
        import platform

        import vhh_library._abnativ_compat as compat_mod

        compat_mod._PATCHED = False
        original_uname = os.uname

        try:
            del os.uname  # type: ignore[attr-defined]
            compat_mod.patch_abnativ_platform()

            result = os.uname()
            assert result.sysname == platform.system()
        finally:
            os.uname = original_uname  # type: ignore[attr-defined]
            compat_mod._PATCHED = False

    def test_patch_is_idempotent(self) -> None:
        """Calling patch twice does not raise or double-patch."""
        import vhh_library._abnativ_compat as compat_mod

        compat_mod._PATCHED = False
        compat_mod.patch_abnativ_platform()
        compat_mod.patch_abnativ_platform()  # second call should be a no-op
        assert compat_mod._PATCHED is True


class TestAbnativInitCli:
    """Verify the vhh-init CLI wrapper applies the patch."""

    def test_main_applies_patch_before_abnativ(self) -> None:
        """vhh-init should call patch_abnativ_platform before abnativ CLI."""
        import vhh_library._abnativ_compat as compat_mod

        patch_called = False
        original_patch = compat_mod.patch_abnativ_platform

        def tracking_patch() -> None:
            nonlocal patch_called
            patch_called = True
            original_patch()

        with mock.patch.object(compat_mod, "patch_abnativ_platform", tracking_patch):
            with mock.patch("abnativ.__main__.main", side_effect=SystemExit(0)):
                with pytest.raises(SystemExit):
                    from vhh_library._abnativ_init_cli import main

                    main()

        assert patch_called
