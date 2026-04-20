"""Tests for vhh_library.runtime_config — config parsing, defaults, fallback."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from vhh_library.runtime_config import (
    VALID_DEVICES,
    VALID_NATIVENESS_BACKENDS,
    VALID_STABILITY_BACKENDS,
    RuntimeConfig,
    resolve_device,
)

# ---------------------------------------------------------------------------
# RuntimeConfig — defaults
# ---------------------------------------------------------------------------


class TestRuntimeConfigDefaults:
    """The default config uses NanoMelt stability + AbNatiV nativeness on auto device."""

    def test_default_device(self):
        cfg = RuntimeConfig()
        assert cfg.device == "auto"

    def test_default_stability_backend(self):
        cfg = RuntimeConfig()
        assert cfg.stability_backend == "nanomelt"

    def test_default_nativeness_backend(self):
        cfg = RuntimeConfig()
        assert cfg.nativeness_backend == "abnativ"

    def test_default_batch_size_is_none(self):
        cfg = RuntimeConfig()
        assert cfg.batch_size is None

    def test_default_cache_dir_is_none(self):
        cfg = RuntimeConfig()
        assert cfg.cache_dir is None

    def test_default_verbose_false(self):
        cfg = RuntimeConfig()
        assert cfg.verbose is False

    def test_default_debug_false(self):
        cfg = RuntimeConfig()
        assert cfg.debug is False


# ---------------------------------------------------------------------------
# RuntimeConfig — validation
# ---------------------------------------------------------------------------


class TestRuntimeConfigValidation:
    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="Invalid device"):
            RuntimeConfig(device="tpu")

    def test_invalid_stability_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid stability_backend"):
            RuntimeConfig(stability_backend="unknown")

    def test_invalid_nativeness_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid nativeness_backend"):
            RuntimeConfig(nativeness_backend="unknown")

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            RuntimeConfig(batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            RuntimeConfig(batch_size=-5)

    def test_valid_devices_accepted(self):
        for d in VALID_DEVICES:
            cfg = RuntimeConfig(device=d)
            assert cfg.device == d

    def test_valid_stability_backends_accepted(self):
        for b in VALID_STABILITY_BACKENDS:
            cfg = RuntimeConfig(stability_backend=b)
            assert cfg.stability_backend == b

    def test_valid_nativeness_backends_accepted(self):
        for b in VALID_NATIVENESS_BACKENDS:
            cfg = RuntimeConfig(nativeness_backend=b)
            assert cfg.nativeness_backend == b

    def test_batch_size_positive_accepted(self):
        cfg = RuntimeConfig(batch_size=32)
        assert cfg.batch_size == 32


# ---------------------------------------------------------------------------
# RuntimeConfig — frozen / immutable
# ---------------------------------------------------------------------------


class TestRuntimeConfigImmutability:
    def test_frozen(self):
        cfg = RuntimeConfig()
        with pytest.raises(AttributeError):
            cfg.device = "cpu"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RuntimeConfig.from_env
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_defaults_when_no_env_vars(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg == RuntimeConfig()

    def test_device_from_env(self):
        with mock.patch.dict(os.environ, {"VHH_DEVICE": "cpu"}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.device == "cpu"

    def test_stability_backend_from_env(self):
        with mock.patch.dict(os.environ, {"VHH_STABILITY_BACKEND": "nanomelt"}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.stability_backend == "nanomelt"

    def test_nativeness_backend_from_env(self):
        with mock.patch.dict(os.environ, {"VHH_NATIVENESS_BACKEND": "abnativ"}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.nativeness_backend == "abnativ"

    def test_batch_size_from_env(self):
        with mock.patch.dict(os.environ, {"VHH_BATCH_SIZE": "64"}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.batch_size == 64

    def test_cache_dir_from_env(self):
        with mock.patch.dict(os.environ, {"VHH_CACHE_DIR": "/tmp/vhh_cache"}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.cache_dir == "/tmp/vhh_cache"

    def test_verbose_true_from_env(self):
        for val in ("1", "true", "yes", "True", "YES"):
            with mock.patch.dict(os.environ, {"VHH_VERBOSE": val}, clear=True):
                cfg = RuntimeConfig.from_env()
            assert cfg.verbose is True, f"Expected verbose=True for VHH_VERBOSE={val!r}"

    def test_verbose_false_from_env(self):
        for val in ("0", "false", "no"):
            with mock.patch.dict(os.environ, {"VHH_VERBOSE": val}, clear=True):
                cfg = RuntimeConfig.from_env()
            assert cfg.verbose is False, f"Expected verbose=False for VHH_VERBOSE={val!r}"

    def test_debug_from_env(self):
        with mock.patch.dict(os.environ, {"VHH_DEBUG": "1"}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.debug is True

    def test_invalid_env_device_raises(self):
        with mock.patch.dict(os.environ, {"VHH_DEVICE": "tpu"}, clear=True):
            with pytest.raises(ValueError, match="Invalid device"):
                RuntimeConfig.from_env()

    def test_invalid_batch_size_warns_and_ignores(self):
        with mock.patch.dict(os.environ, {"VHH_BATCH_SIZE": "not_a_number"}, clear=True):
            with pytest.warns(RuntimeWarning, match="not a valid integer"):
                cfg = RuntimeConfig.from_env()
        assert cfg.batch_size is None

    def test_whitespace_trimmed(self):
        with mock.patch.dict(os.environ, {"VHH_DEVICE": "  cpu  "}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.device == "cpu"

    def test_case_insensitive(self):
        with mock.patch.dict(os.environ, {"VHH_DEVICE": "CPU"}, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.device == "cpu"

    def test_multiple_env_vars(self):
        env = {
            "VHH_DEVICE": "cuda",
            "VHH_STABILITY_BACKEND": "both",
            "VHH_BATCH_SIZE": "128",
            "VHH_VERBOSE": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = RuntimeConfig.from_env()
        assert cfg.device == "cuda"
        assert cfg.stability_backend == "both"
        assert cfg.batch_size == 128
        assert cfg.verbose is True


# ---------------------------------------------------------------------------
# RuntimeConfig.from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_empty_dict_gives_defaults(self):
        cfg = RuntimeConfig.from_dict({})
        assert cfg == RuntimeConfig()

    def test_partial_dict(self):
        cfg = RuntimeConfig.from_dict({"device": "cpu", "batch_size": 16})
        assert cfg.device == "cpu"
        assert cfg.batch_size == 16
        assert cfg.stability_backend == "nanomelt"  # default

    def test_unknown_keys_ignored(self):
        cfg = RuntimeConfig.from_dict({"device": "cpu", "unknown_key": 42})
        assert cfg.device == "cpu"

    def test_full_dict(self):
        data = {
            "device": "cuda",
            "stability_backend": "nanomelt",
            "nativeness_backend": "abnativ",
            "batch_size": 256,
            "cache_dir": "/tmp/cache",
            "verbose": True,
            "debug": False,
        }
        cfg = RuntimeConfig.from_dict(data)
        assert cfg.device == "cuda"
        assert cfg.stability_backend == "nanomelt"
        assert cfg.batch_size == 256
        assert cfg.cache_dir == "/tmp/cache"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


class TestConvenienceHelpers:
    def test_needs_esm2_for_esm2_backend(self):
        assert RuntimeConfig(stability_backend="esm2").needs_esm2() is True

    def test_needs_esm2_for_both_backend(self):
        assert RuntimeConfig(stability_backend="both").needs_esm2() is True

    def test_needs_esm2_false_for_nanomelt(self):
        assert RuntimeConfig(stability_backend="nanomelt").needs_esm2() is False

    def test_needs_nanomelt_for_nanomelt_backend(self):
        assert RuntimeConfig(stability_backend="nanomelt").needs_nanomelt() is True

    def test_needs_nanomelt_for_both_backend(self):
        assert RuntimeConfig(stability_backend="both").needs_nanomelt() is True

    def test_needs_nanomelt_false_for_esm2(self):
        assert RuntimeConfig(stability_backend="esm2").needs_nanomelt() is False

    def test_configure_logging_debug(self):
        import logging

        cfg = RuntimeConfig(debug=True)
        cfg.configure_logging()
        pkg_logger = logging.getLogger("vhh_library")
        assert pkg_logger.level == logging.DEBUG

    def test_configure_logging_verbose(self):
        import logging

        cfg = RuntimeConfig(verbose=True)
        cfg.configure_logging()
        pkg_logger = logging.getLogger("vhh_library")
        assert pkg_logger.level == logging.INFO

    def test_configure_logging_debug_takes_precedence(self):
        import logging

        cfg = RuntimeConfig(verbose=True, debug=True)
        cfg.configure_logging()
        pkg_logger = logging.getLogger("vhh_library")
        assert pkg_logger.level == logging.DEBUG


# ---------------------------------------------------------------------------
# resolve_device (module-level function)
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert resolve_device("cpu") == "cpu"

    def test_auto_without_torch_returns_cpu(self):
        with mock.patch.dict("sys.modules", {"torch": None}):
            assert resolve_device("auto") == "cpu"

    def test_auto_with_cuda(self):
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device("auto") == "cuda"

    def test_auto_with_mps(self):
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device("auto") == "mps"

    def test_auto_cpu_fallback(self):
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            assert resolve_device("auto") == "cpu"

    def test_cuda_unavailable_falls_back_to_cpu(self):
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with pytest.warns(RuntimeWarning, match="CUDA requested but not available"):
                result = resolve_device("cuda")
        assert result == "cpu"

    def test_cuda_without_torch_falls_back_to_cpu(self):
        with mock.patch.dict("sys.modules", {"torch": None}):
            with pytest.warns(RuntimeWarning, match="torch is not installed"):
                result = resolve_device("cuda")
        assert result == "cpu"

    def test_mps_unavailable_falls_back_to_cpu(self):
        mock_torch = mock.MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with pytest.warns(RuntimeWarning, match="MPS requested but not available"):
                result = resolve_device("mps")
        assert result == "cpu"

    def test_mps_without_torch_falls_back_to_cpu(self):
        with mock.patch.dict("sys.modules", {"torch": None}):
            with pytest.warns(RuntimeWarning, match="torch is not installed"):
                result = resolve_device("mps")
        assert result == "cpu"

    def test_auto_cuda_smoke_test_failure_falls_back_to_cpu(self):
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.zeros.side_effect = RuntimeError("cublasLtCreate")
        mock_torch.backends.mps.is_available.return_value = False
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with pytest.warns(RuntimeWarning, match="failed a smoke test"):
                result = resolve_device("auto")
        assert result == "cpu"

    def test_cuda_smoke_test_failure_falls_back_to_cpu(self):
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.zeros.side_effect = RuntimeError("cublasLtCreate")
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with pytest.warns(RuntimeWarning, match="failed a smoke test"):
                result = resolve_device("cuda")
        assert result == "cpu"

    def test_unknown_device_passed_through(self):
        # Unknown strings are passed through — torch will raise if invalid
        assert resolve_device("xla") == "xla"

    def test_resolve_device_on_config_instance(self):
        """RuntimeConfig.resolve_device delegates to the module function."""
        cfg = RuntimeConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"
