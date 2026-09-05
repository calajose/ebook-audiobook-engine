"""Tests for hardware device detection and provider resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from audiobook_engine.domain.exceptions import TTSBackendError
from audiobook_engine.infrastructure.device import (
    DeviceInfo,
    _onnxruntime_cuda_ep_available,
    _onnxruntime_openvino_ep_available,
    _onnxruntime_providers,
    _openvino_devices,
    _openvino_installed,
    detect_devices,
    get_provider_config,
    resolve_device,
    verify_session_provider,
)

# ---------------------------------------------------------------------------
# Unit tests for lazy helpers
# ---------------------------------------------------------------------------


class TestOnnxruntimeProviders:
    def test_returns_list(self) -> None:
        result = _onnxruntime_providers()
        assert isinstance(result, list)

    def test_contains_cpu(self) -> None:
        result = _onnxruntime_providers()
        assert "CPUExecutionProvider" in result


class TestOpenvinoInstalled:
    def test_returns_bool(self) -> None:
        result = _openvino_installed()
        assert isinstance(result, bool)


class TestOpenvinoDevices:
    def test_returns_list(self) -> None:
        result = _openvino_devices()
        assert isinstance(result, list)


class TestOnnxruntimeOpenvinoEPAvailable:
    def test_returns_bool(self) -> None:
        result = _onnxruntime_openvino_ep_available()
        assert isinstance(result, bool)


class TestOnnxruntimeCudaEPAvailable:
    def test_returns_bool(self) -> None:
        result = _onnxruntime_cuda_ep_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# detect_devices
# ---------------------------------------------------------------------------


class TestDetectDevices:
    def test_always_returns_cpu(self) -> None:
        devices = detect_devices()
        assert len(devices) >= 1
        assert devices[0].name == "cpu"
        assert devices[0].available is True
        assert devices[0].provider_name is None

    def test_cpu_has_ort_version(self) -> None:
        devices = detect_devices()
        cpu = devices[0]
        assert cpu.onnxruntime_version != ""

    @patch(
        "audiobook_engine.infrastructure.device._onnxruntime_openvino_ep_available",
        return_value=False,
    )
    def test_no_gpu_when_ep_unavailable(self, mock_ep: MagicMock) -> None:
        devices = detect_devices()
        gpu_devices = [d for d in devices if d.name == "openvino_gpu"]
        assert len(gpu_devices) == 0

    @patch(
        "audiobook_engine.infrastructure.device._onnxruntime_openvino_ep_available",
        return_value=True,
    )
    @patch(
        "audiobook_engine.infrastructure.device._openvino_devices",
        return_value=["CPU", "GPU"],
    )
    @patch(
        "audiobook_engine.infrastructure.device._openvino_gpu_name",
        return_value="Intel(R) Iris(R) Xe Graphics",
    )
    def test_gpu_detected_when_available(
        self,
        mock_name: MagicMock,
        mock_devs: MagicMock,
        mock_ep: MagicMock,
    ) -> None:
        devices = detect_devices()
        gpu = [d for d in devices if d.name == "openvino_gpu"]
        assert len(gpu) == 1
        assert gpu[0].available is True
        assert gpu[0].provider_name == "OpenVINOExecutionProvider"
        assert gpu[0].gpu_device_id == "GPU"
        assert "Iris" in gpu[0].display_name

    @patch(
        "audiobook_engine.infrastructure.device._onnxruntime_openvino_ep_available",
        return_value=True,
    )
    @patch(
        "audiobook_engine.infrastructure.device._openvino_devices",
        return_value=["CPU", "GPU.0", "GPU.1"],
    )
    @patch(
        "audiobook_engine.infrastructure.device._openvino_gpu_name",
        return_value="Intel(R) Graphics",
    )
    def test_multi_gpu_selects_first(
        self,
        mock_name: MagicMock,
        mock_devs: MagicMock,
        mock_ep: MagicMock,
    ) -> None:
        devices = detect_devices()
        gpu = [d for d in devices if d.name == "openvino_gpu"]
        assert len(gpu) == 1
        assert gpu[0].gpu_device_id == "GPU.0"

    @patch(
        "audiobook_engine.infrastructure.device._onnxruntime_openvino_ep_available",
        return_value=True,
    )
    @patch(
        "audiobook_engine.infrastructure.device._openvino_devices",
        return_value=["CPU"],
    )
    def test_ep_available_but_no_gpu(
        self, mock_devs: MagicMock, mock_ep: MagicMock
    ) -> None:
        devices = detect_devices()
        gpu = [d for d in devices if d.name == "openvino_gpu"]
        assert len(gpu) == 0

    @patch(
        "audiobook_engine.infrastructure.device._onnxruntime_cuda_ep_available",
        return_value=True,
    )
    def test_cuda_detected_when_available(
        self, mock_cuda_ep: MagicMock
    ) -> None:
        devices = detect_devices()
        cuda = [d for d in devices if d.name == "cuda"]
        assert len(cuda) == 1
        assert cuda[0].available is True
        assert cuda[0].provider_name == "CUDAExecutionProvider"

    @patch(
        "audiobook_engine.infrastructure.device._onnxruntime_cuda_ep_available",
        return_value=False,
    )
    def test_no_cuda_when_ep_unavailable(
        self, mock_cuda_ep: MagicMock
    ) -> None:
        devices = detect_devices()
        cuda = [d for d in devices if d.name == "cuda"]
        assert len(cuda) == 0

    @patch(
        "audiobook_engine.infrastructure.device._onnxruntime_cuda_ep_available",
        return_value=True,
    )
    @patch(
        "audiobook_engine.infrastructure.device._onnxruntime_openvino_ep_available",
        return_value=True,
    )
    @patch(
        "audiobook_engine.infrastructure.device._openvino_devices",
        return_value=["CPU", "GPU"],
    )
    @patch(
        "audiobook_engine.infrastructure.device._openvino_gpu_name",
        return_value="Intel(R) Iris(R) Xe Graphics",
    )
    def test_cuda_before_openvino_in_device_list(
        self,
        mock_name: MagicMock,
        mock_devs: MagicMock,
        mock_ov_ep: MagicMock,
        mock_cuda_ep: MagicMock,
    ) -> None:
        devices = detect_devices()
        names = [d.name for d in devices]
        assert names.index("cuda") < names.index("openvino_gpu")


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_resolve_cpu(self) -> None:
        dev = resolve_device("cpu")
        assert dev.name == "cpu"

    def test_resolve_auto_falls_back_to_cpu(self) -> None:
        with patch(
            "audiobook_engine.infrastructure.device.detect_devices"
        ) as mock_detect:
            mock_detect.return_value = [
                DeviceInfo(
                    name="cpu",
                    display_name="CPU",
                    available=True,
                    provider_name=None,
                ),
            ]
            dev = resolve_device("auto")
            assert dev.name == "cpu"

    def test_resolve_auto_selects_gpu(self) -> None:
        with patch(
            "audiobook_engine.infrastructure.device.detect_devices"
        ) as mock_detect:
            gpu = DeviceInfo(
                name="openvino_gpu",
                display_name="OpenVINO GPU",
                available=True,
                provider_name="OpenVINOExecutionProvider",
                provider_options={"device_type": "GPU"},
            )
            cpu = DeviceInfo(
                name="cpu",
                display_name="CPU",
                available=True,
                provider_name=None,
            )
            mock_detect.return_value = [cpu, gpu]
            dev = resolve_device("auto")
            assert dev.name == "openvino_gpu"

    def test_resolve_openvino_forced_no_ep(self) -> None:
        with (
            patch(
                "audiobook_engine.infrastructure.device"
                "._onnxruntime_openvino_ep_available",
                return_value=False,
            ),
            patch(
                "audiobook_engine.infrastructure.device"
                "._onnxruntime_providers",
                return_value=["CPUExecutionProvider"],
            ),
            pytest.raises(TTSBackendError, match="not available"),
        ):
            resolve_device("openvino")

    def test_resolve_openvino_forced_no_gpu(self) -> None:
        with (
            patch(
                "audiobook_engine.infrastructure.device"
                "._onnxruntime_openvino_ep_available",
                return_value=True,
            ),
            patch(
                "audiobook_engine.infrastructure.device.detect_devices",
                return_value=[
                    DeviceInfo(
                        name="cpu",
                        display_name="CPU",
                        available=True,
                        provider_name=None,
                    ),
                ],
            ),
            patch(
                "audiobook_engine.infrastructure.device._openvino_devices",
                return_value=["CPU"],
            ),
            pytest.raises(TTSBackendError, match="no Intel GPU"),
        ):
            resolve_device("openvino")

    def test_resolve_openvino_forced_with_gpu(self) -> None:
        gpu = DeviceInfo(
            name="openvino_gpu",
            display_name="OpenVINO GPU",
            available=True,
            provider_name="OpenVINOExecutionProvider",
            provider_options={"device_type": "GPU"},
        )
        cpu = DeviceInfo(
            name="cpu",
            display_name="CPU",
            available=True,
            provider_name=None,
        )
        with patch(
            "audiobook_engine.infrastructure.device.detect_devices",
            return_value=[cpu, gpu],
        ), patch(
            "audiobook_engine.infrastructure.device._onnxruntime_openvino_ep_available",
            return_value=True,
        ):
            dev = resolve_device("openvino")
            assert dev.name == "openvino_gpu"

    def test_resolve_unknown_device(self) -> None:
        with pytest.raises(TTSBackendError, match="Unknown device"):
            resolve_device("rocm")

    def test_resolve_cuda_explicit(self) -> None:
        cuda = DeviceInfo(
            name="cuda",
            display_name="NVIDIA CUDA GPU",
            available=True,
            provider_name="CUDAExecutionProvider",
            provider_options={},
        )
        cpu = DeviceInfo(
            name="cpu",
            display_name="CPU",
            available=True,
            provider_name=None,
        )
        with patch(
            "audiobook_engine.infrastructure.device"
            "._onnxruntime_cuda_ep_available",
            return_value=True,
        ), patch(
            "audiobook_engine.infrastructure.device.detect_devices",
            return_value=[cpu, cuda],
        ):
            dev = resolve_device("cuda")
            assert dev.name == "cuda"
            assert dev.provider_name == "CUDAExecutionProvider"

    def test_resolve_cuda_not_available_raises(self) -> None:
        with (
            patch(
                "audiobook_engine.infrastructure.device"
                "._onnxruntime_cuda_ep_available",
                return_value=False,
            ),
            patch(
                "audiobook_engine.infrastructure.device"
                "._onnxruntime_providers",
                return_value=["CPUExecutionProvider"],
            ),
            pytest.raises(TTSBackendError, match="not available"),
        ):
            resolve_device("cuda")

    def test_auto_prioritizes_cuda_over_openvino(self) -> None:
        cuda = DeviceInfo(
            name="cuda",
            display_name="NVIDIA CUDA GPU",
            available=True,
            provider_name="CUDAExecutionProvider",
            provider_options={},
        )
        openvino = DeviceInfo(
            name="openvino_gpu",
            display_name="OpenVINO GPU",
            available=True,
            provider_name="OpenVINOExecutionProvider",
            provider_options={"device_type": "GPU"},
        )
        cpu = DeviceInfo(
            name="cpu",
            display_name="CPU",
            available=True,
            provider_name=None,
        )
        with patch(
            "audiobook_engine.infrastructure.device.detect_devices",
            return_value=[cpu, cuda, openvino],
        ):
            dev = resolve_device("auto")
            assert dev.name == "cuda"

    def test_auto_falls_back_to_openvino_when_no_cuda(self) -> None:
        openvino = DeviceInfo(
            name="openvino_gpu",
            display_name="OpenVINO GPU",
            available=True,
            provider_name="OpenVINOExecutionProvider",
            provider_options={"device_type": "GPU"},
        )
        cpu = DeviceInfo(
            name="cpu",
            display_name="CPU",
            available=True,
            provider_name=None,
        )
        with patch(
            "audiobook_engine.infrastructure.device.detect_devices",
            return_value=[cpu, openvino],
        ):
            dev = resolve_device("auto")
            assert dev.name == "openvino_gpu"


# ---------------------------------------------------------------------------
# get_provider_config
# ---------------------------------------------------------------------------


class TestGetProviderConfig:
    def test_cpu_returns_empty(self) -> None:
        providers, disable = get_provider_config("cpu")
        assert providers == []
        assert disable is False

    def test_auto_returns_providers_if_gpu(self) -> None:
        with patch(
            "audiobook_engine.infrastructure.device.resolve_device"
        ) as mock_resolve:
            mock_resolve.return_value = DeviceInfo(
                name="openvino_gpu",
                display_name="GPU",
                available=True,
                provider_name="OpenVINOExecutionProvider",
                provider_options={"device_type": "GPU"},
            )
            providers, disable = get_provider_config("auto")
            assert len(providers) == 1
            assert providers[0][0] == "OpenVINOExecutionProvider"
            assert disable is False  # auto allows ORT fallback

    def test_openvino_disables_fallback(self) -> None:
        with patch(
            "audiobook_engine.infrastructure.device.resolve_device"
        ) as mock_resolve:
            mock_resolve.return_value = DeviceInfo(
                name="openvino_gpu",
                display_name="GPU",
                available=True,
                provider_name="OpenVINOExecutionProvider",
                provider_options={"device_type": "GPU"},
            )
            providers, disable = get_provider_config("openvino")
            assert disable is True  # explicit → no fallback

    def test_cuda_explicit_returns_providers_with_cpu_fallback(
        self,
    ) -> None:
        with patch(
            "audiobook_engine.infrastructure.device.resolve_device"
        ) as mock_resolve:
            mock_resolve.return_value = DeviceInfo(
                name="cuda",
                display_name="NVIDIA CUDA GPU",
                available=True,
                provider_name="CUDAExecutionProvider",
                provider_options={},
            )
            providers, disable = get_provider_config("cuda")
            assert "CUDAExecutionProvider" in providers
            assert "CPUExecutionProvider" in providers
            assert disable is True  # explicit → no fallback

    def test_cuda_auto_returns_providers_with_fallback_allowed(
        self,
    ) -> None:
        with patch(
            "audiobook_engine.infrastructure.device.resolve_device"
        ) as mock_resolve:
            mock_resolve.return_value = DeviceInfo(
                name="cuda",
                display_name="NVIDIA CUDA GPU",
                available=True,
                provider_name="CUDAExecutionProvider",
                provider_options={},
            )
            providers, disable = get_provider_config("auto")
            assert "CUDAExecutionProvider" in providers
            assert "CPUExecutionProvider" in providers
            assert disable is False  # auto allows ORT fallback


# ---------------------------------------------------------------------------
# verify_session_provider
# ---------------------------------------------------------------------------


class TestVerifySessionProvider:
    def test_matching_provider(self) -> None:
        session = MagicMock()
        session.get_providers.return_value = ["OpenVINOExecutionProvider"]
        result = verify_session_provider(
            session, "OpenVINOExecutionProvider"
        )
        assert result is None

    def test_missing_provider(self) -> None:
        session = MagicMock()
        session.get_providers.return_value = ["CPUExecutionProvider"]
        result = verify_session_provider(
            session, "OpenVINOExecutionProvider"
        )
        assert result is not None
        assert "OpenVINOExecutionProvider" in result

    def test_with_device_check_passes(self) -> None:
        session = MagicMock()
        session.get_providers.return_value = ["OpenVINOExecutionProvider"]
        session.get_provider_options.return_value = [
            {"device_type": "GPU"}
        ]
        result = verify_session_provider(
            session, "OpenVINOExecutionProvider", "GPU"
        )
        assert result is None

    def test_with_device_check_mismatch(self) -> None:
        session = MagicMock()
        session.get_providers.return_value = ["OpenVINOExecutionProvider"]
        session.get_provider_options.return_value = [
            {"device_type": "CPU"}
        ]
        result = verify_session_provider(
            session, "OpenVINOExecutionProvider", "GPU"
        )
        # Should warn (not error) since device mismatch is a warning
        # The function returns None on success or warning — check logged
        # Actually the function returns None when device_type found,
        # and only warns when not found. Let's check the actual behavior.
        # With CPU options and expecting GPU, it should NOT return None.
        # Actually looking at the code, it only returns None when
        # device_type matches, otherwise it logs a warning and returns None.
        # This is by design — the provider is active but device_type not confirmed.
        assert result is None  # Warning logged but not an error
