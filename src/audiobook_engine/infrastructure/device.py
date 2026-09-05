"""Hardware device detection and ONNX Runtime provider resolution.

Centralizes detection of available compute backends (CPU, CUDA NVIDIA GPU,
OpenVINO GPU) and maps user-facing device names to ONNX Runtime execution
provider configurations.

This module is self-contained: every external import is lazy so the
rest of the application never fails because an optional backend
package is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    """Describes a detected compute device."""

    name: str
    """Internal identifier: ``"cpu"``, ``"cuda"``, ``"openvino_gpu"``."""

    display_name: str
    """Human-readable label shown in diagnostics: ``"CPU"``,
    ``"OpenVINO GPU — Intel(R) Iris(R) Xe Graphics"``."""

    available: bool
    """Whether the device can actually be used right now."""

    provider_name: str | None
    """ONNX Runtime execution-provider name, e.g.
    ``"OpenVINOExecutionProvider"``.  ``None`` for plain CPU."""

    provider_options: dict[str, Any] = field(default_factory=dict)
    """Options forwarded to ``ort.InferenceSession(providers=[…])``."""

    openvino_version: str | None = None
    """OpenVINO version string, or ``None`` if unavailable."""

    onnxruntime_version: str = ""
    """ONNX Runtime version string."""

    openvino_devices: list[str] = field(default_factory=list)
    """Devices reported by ``openvino.Core().available_devices``."""

    gpu_device_id: str | None = None
    """OpenVINO device identifier for the GPU, e.g. ``"GPU"`` or
    ``"GPU.0"``.  ``None`` when no GPU is present."""


# ---------------------------------------------------------------------------
# Lazy helpers — every import is guarded so missing packages are never fatal
# ---------------------------------------------------------------------------


def _onnxruntime_version() -> str:
    try:
        import onnxruntime as _ort

        ver: str = _ort.__version__
        return ver
    except ImportError:
        return ""


def _onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime as _ort

        provs: list[str] = _ort.get_available_providers()
        return provs
    except ImportError:
        return []


def _openvino_installed() -> bool:
    try:
        import openvino as _ov  # noqa: F401

        return True
    except ImportError:
        return False


def _openvino_version() -> str | None:
    try:
        import openvino as _ov

        ver: str = _ov.__version__
        return ver
    except (ImportError, AttributeError):
        return None


def _openvino_devices() -> list[str]:
    """Return list of device identifiers from OpenVINO Core.

    On a system with Intel Iris Xe this is typically ``["CPU", "GPU"]``.
    """
    try:
        import openvino as _ov

        core = _ov.Core()
        devs: list[str] = list(core.available_devices)
        return devs
    except Exception:
        return []


def _openvino_gpu_name(device_id: str = "GPU") -> str | None:
    """Return the full device name for an OpenVINO GPU, or ``None``."""
    try:
        import openvino as _ov

        core = _ov.Core()
        name: str = core.get_property(device_id, "FULL_DEVICE_NAME")
        return name
    except Exception:
        return None


def _onnxruntime_cuda_ep_available() -> bool:
    """Check if ``CUDAExecutionProvider`` is in ORT's provider list."""
    return "CUDAExecutionProvider" in _onnxruntime_providers()


def _onnxruntime_openvino_ep_available() -> bool:
    """Check if ``OpenVINOExecutionProvider`` is in ORT's provider list."""
    return "OpenVINOExecutionProvider" in _onnxruntime_providers()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_devices() -> list[DeviceInfo]:
    """Detect all available compute devices.

    Returns a list with at least a CPU entry.  GPU entries are appended
    only when the corresponding libraries are installed AND the hardware
    is present.
    """
    ort_version = _onnxruntime_version()
    ov_version = _openvino_version()
    devices: list[DeviceInfo] = []

    # --- CPU (always available) ---
    devices.append(
        DeviceInfo(
            name="cpu",
            display_name="CPU",
            available=True,
            provider_name=None,
            provider_options={},
            openvino_version=ov_version,
            onnxruntime_version=ort_version,
        )
    )

    # --- CUDA (NVIDIA GPU) ---
    cuda_ep_ready = _onnxruntime_cuda_ep_available()
    if cuda_ep_ready:
        devices.append(
            DeviceInfo(
                name="cuda",
                display_name="NVIDIA CUDA GPU",
                available=True,
                provider_name="CUDAExecutionProvider",
                provider_options={},
                onnxruntime_version=ort_version,
            )
        )

    # --- OpenVINO GPU ---
    ov_ep_ready = _onnxruntime_openvino_ep_available()
    ov_devices = _openvino_devices()
    gpu_ids = [d for d in ov_devices if d.startswith("GPU")]

    if ov_ep_ready and gpu_ids:
        gpu_id = gpu_ids[0]  # prefer GPU.0 or GPU
        gpu_name = _openvino_gpu_name(gpu_id) or gpu_id
        devices.append(
            DeviceInfo(
                name="openvino_gpu",
                display_name=f"OpenVINO {gpu_id} — {gpu_name}",
                available=True,
                provider_name="OpenVINOExecutionProvider",
                provider_options={"device_type": gpu_id},
                openvino_version=ov_version,
                onnxruntime_version=ort_version,
                openvino_devices=ov_devices,
                gpu_device_id=gpu_id,
            )
        )
    elif ov_ep_ready:
        # EP is registered but no GPU device found — still useful for
        # CPUviaOpenVINO (not exposed as a user-facing option yet).
        log.debug(
            "OpenVINOExecutionProvider available but no GPU device detected"
        )

    return devices


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_device(device: str) -> DeviceInfo:
    """Resolve a user-facing device name to a ``DeviceInfo``.

    Parameters
    ----------
    device:
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"openvino"``.

    Returns
    -------
    DeviceInfo
        The best matching device.

    Raises
    ------
    TTSBackendError
        If *device* is ``"cuda"`` or ``"openvino"`` and the backend
        is not usable.
    """
    from audiobook_engine.domain.exceptions import TTSBackendError

    devices = detect_devices()

    if device == "cpu":
        cpu = devices[0]  # CPU is always first
        log.debug("Device forced to CPU")
        return cpu

    if device == "cuda":
        return _resolve_cuda_forced(devices)

    if device == "openvino":
        return _resolve_openvino_forced(devices)

    if device == "auto":
        return _resolve_auto(devices)

    raise TTSBackendError(
        f"Unknown device '{device}'. Valid options: auto, cpu, cuda, openvino"
    )


def _resolve_auto(devices: list[DeviceInfo]) -> DeviceInfo:
    """Auto-select the best available device (CUDA > OpenVINO GPU > CPU)."""
    for dev in devices:
        if dev.name == "cuda" and dev.available:
            log.info("Auto-selected device: %s", dev.display_name)
            return dev

    for dev in devices:
        if dev.name == "openvino_gpu" and dev.available:
            log.info("Auto-selected device: %s", dev.display_name)
            return dev

    cpu = devices[0]
    log.info("Auto-selected device: %s (no GPU available)", cpu.display_name)
    return cpu


def _resolve_cuda_forced(devices: list[DeviceInfo]) -> DeviceInfo:
    """Force CUDA GPU — raise a clear error if not usable."""
    from audiobook_engine.domain.exceptions import TTSBackendError

    if not _onnxruntime_cuda_ep_available():
        available = _onnxruntime_providers()
        raise TTSBackendError(
            "CUDA Execution Provider is not available.\n"
            "Install with: pip install 'ebook-audiobook-engine[cuda]'\n"
            "Ensure NVIDIA GPU drivers and CUDA toolkit are installed.\n"
            f"Currently available ORT providers: {available}"
        )

    for dev in devices:
        if dev.name == "cuda" and dev.available:
            log.info("Forced device: %s", dev.display_name)
            return dev

    # EP was reported available but device not found — should not happen
    # for CUDA (no separate GPU device enumeration needed like OpenVINO).
    raise TTSBackendError(
        "CUDA Execution Provider is registered but no CUDA device "
        "was detected. Ensure your NVIDIA GPU is functional."
    )


def _resolve_openvino_forced(devices: list[DeviceInfo]) -> DeviceInfo:
    """Force OpenVINO GPU — raise a clear error if not usable."""
    from audiobook_engine.domain.exceptions import TTSBackendError

    # Check 1: onnxruntime-openvino installed?
    if not _onnxruntime_openvino_ep_available():
        available = _onnxruntime_providers()
        raise TTSBackendError(
            "OpenVINO Execution Provider is not available.\n"
            "Install with: pip install 'ebook-audiobook-engine[openvino]'\n"
            f"Currently available ORT providers: {available}"
        )

    # Check 2: OpenVINO GPU detected?
    for dev in devices:
        if dev.name == "openvino_gpu" and dev.available:
            log.info("Forced device: %s", dev.display_name)
            return dev

    ov_devices = _openvino_devices()
    raise TTSBackendError(
        "OpenVINO is installed but no Intel GPU was detected.\n"
        f"OpenVINO devices found: {ov_devices}\n"
        "Ensure Intel GPU drivers and Compute Runtime are installed.\n"
        "See: https://docs.openvino.ai/2024/get-started/configurations/"
        "configurations-intel-gpu.html"
    )


def get_provider_config(
    device: str,
) -> tuple[list[str | tuple[str, dict[str, Any]]], bool]:
    """Return ``(providers, disable_fallback)`` for ``ort.InferenceSession``.

    Parameters
    ----------
    device:
        User-facing device name.

    Returns
    -------
    providers:
        List of ``(provider_name, options)`` tuples, or list of provider
        name strings.  Empty list means plain CPU with no EP override.
    disable_fallback:
        ``True`` when the user explicitly requested a specific backend
        and ORT must NOT silently fall back to CPU.
    """
    resolved = resolve_device(device)

    if resolved.provider_name is None:
        # Plain CPU — no provider options needed
        return ([], False)

    if resolved.provider_name == "CUDAExecutionProvider":
        # CUDA: include CPU as fallback for unsupported operators,
        # but prevent ORT from silently falling back at runtime.
        providers: list[str | tuple[str, dict[str, Any]]] = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        disable_fallback = device != "auto"
        return (providers, disable_fallback)

    providers_config: list[str | tuple[str, dict[str, Any]]] = [
        (resolved.provider_name, resolved.provider_options)
    ]

    # Disable ORT's internal CPU fallback only when the user explicitly
    # requested a non-CPU backend.  In "auto" mode the *application*
    # handles fallback (auto → CPU with warning), but ORT should still
    # raise if its own EP fails during session creation.
    disable_fallback = device != "auto"

    return (providers_config, disable_fallback)


def verify_session_provider(
    session: Any,
    expected_provider: str,
    expected_device: str | None = None,
) -> str | None:
    """Verify that an ORT session is actually using the expected EP.

    Returns ``None`` on success, or a human-readable error string.
    """
    active = session.get_providers()

    if expected_provider not in active:
        return (
            f"Requested {expected_provider} but session is using {active}. "
            f"The EP may have fallen back to CPU."
        )

    if expected_device:
        opts = session.get_provider_options()
        for opt in opts:
            if isinstance(opt, dict) and opt.get("device_type") == expected_device:
                return None
        # device_type not found in options — might still be OK if the
        # EP defaulted to the requested device, but warn.
        log.warning(
            "Provider registered but device_type '%s' not confirmed "
            "in provider options: %s",
            expected_device,
            opts,
        )

    return None
