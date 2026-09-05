#!/usr/bin/env python3
"""Hardware detection diagnostic for ebook-audiobook-engine.

Run standalone to inspect available compute devices, OpenVINO status,
ONNX Runtime providers, and which device Kokoro would select.

Usage:
    python scripts/detect_device.py
    python scripts/detect_device.py --device openvino
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

# Ensure the project source is importable when run from the repo root.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def _section(title: str) -> None:
    print(f"\n{title}:")


def _kv(key: str, value: str) -> None:
    print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS hardware detection")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "openvino"],
        help="Device to test (default: auto)",
    )
    args = parser.parse_args()

    print("TTS Hardware Detection")
    print("=" * 40)

    # --- ONNX Runtime ---
    _section("ONNX Runtime")
    try:
        import onnxruntime as ort

        _kv("version", ort.__version__)
        providers = ort.get_available_providers()
        _kv("available_providers", str(providers))
        _kv(
            "CUDAExecutionProvider",
            "yes" if "CUDAExecutionProvider" in providers else "no",
        )
        _kv(
            "OpenVINOExecutionProvider",
            "yes" if "OpenVINOExecutionProvider" in providers else "no",
        )
    except ImportError:
        _kv("installed", "no")

    # --- onnxruntime-openvino package ---
    _section("onnxruntime-openvino")
    try:
        dist = importlib.metadata.distribution("onnxruntime-openvino")
        _kv("version", dist.version)
    except importlib.metadata.PackageNotFoundError:
        _kv("installed", "no")

    # --- onnxruntime-gpu package ---
    _section("onnxruntime-gpu")
    try:
        dist = importlib.metadata.distribution("onnxruntime-gpu")
        _kv("version", dist.version)
    except importlib.metadata.PackageNotFoundError:
        _kv("installed", "no")

    # --- OpenVINO (standalone Python package) ---
    _section("OpenVINO")
    try:
        import openvino as ov

        _kv("version", ov.__version__)
        core = ov.Core()
        devices = list(core.available_devices)
        _kv("devices", str(devices))
        for dev in devices:
            if dev.startswith("GPU"):
                try:
                    name = core.get_property(dev, "FULL_DEVICE_NAME")
                    _kv(f"  {dev}", name)
                except Exception:
                    _kv(f"  {dev}", "(could not read name)")
    except ImportError:
        _kv("installed", "no (not needed for ONNX Runtime OpenVINO EP on Linux)")

    # --- Device resolution ---
    _section("Device Resolution")
    from audiobook_engine.infrastructure.device import (
        detect_devices,
        resolve_device,
    )

    devices = detect_devices()
    for dev in devices:
        status = "available" if dev.available else "unavailable"
        _kv(dev.display_name, status)
        if dev.openvino_version:
            _kv("  openvino_version", dev.openvino_version)
        if dev.openvino_devices:
            _kv("  openvino_devices", str(dev.openvino_devices))

    _section(f"Resolving --device {args.device}")
    try:
        resolved = resolve_device(args.device)
        _kv("selected", resolved.display_name)
        _kv("provider", resolved.provider_name or "(none)")
        _kv("provider_options", str(resolved.provider_options))
    except Exception as exc:
        _kv("error", str(exc))
        sys.exit(1)

    # --- Kokoro session test (if model available) ---
    _section("Kokoro Session Test")
    try:
        from kokoro_onnx import Kokoro

        from audiobook_engine.infrastructure.tts.kokoro.models import (
            get_model_path,
            get_voices_path,
        )

        model_path = get_model_path()
        voices_path = get_voices_path()
        _kv("model_path", str(model_path))
        _kv("voices_path", str(voices_path))

        if resolved.provider_name:
            import onnxruntime as ort

            if resolved.provider_name == "CUDAExecutionProvider":
                # CUDA uses plain string list with CPU fallback
                providers = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]
            else:
                providers = [
                    (resolved.provider_name, resolved.provider_options)
                ]
            try:
                session = ort.InferenceSession(
                    str(model_path),
                    providers=providers,
                    enable_fallback=0,
                )
                active = session.get_providers()
                _kv("session_providers", str(active))
                _kv(
                    "ep_active",
                    "yes" if resolved.provider_name in active
                    else "NO — fallback!",
                )
                kokoro = Kokoro.from_session(session, str(voices_path))
                _kv("voices", str(kokoro.get_voices()[:5]) + "...")
                _kv("status", "OK")
            except Exception as session_exc:
                err = str(session_exc)
                _kv("session_error", err)
                if "dynamic" in err.lower() or "SequenceAt" in err:
                    _kv(
                        "note",
                        "The Kokoro ONNX model uses operators "
                        "(SplitToSequence, SequenceAt) that OpenVINO "
                        "cannot convert. Use --device cpu instead.",
                    )
        else:
            kokoro = Kokoro(str(model_path), str(voices_path))
            _kv("session_providers", str(kokoro.sess.get_providers()))
            _kv("voices", str(kokoro.get_voices()[:5]) + "...")
            _kv("status", "OK")
    except Exception as exc:
        _kv("error", str(exc))


if __name__ == "__main__":
    main()
