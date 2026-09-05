#!/usr/bin/env python3
"""Benchmark TTS synthesis across compute devices.

Compares CPU vs OpenVINO GPU (or any available device) using the
same model, voice, and text.  Reports inference time, audio duration,
and real-time factor (RTF).

Usage:
    python scripts/benchmark_tts.py --device cpu
    python scripts/benchmark_tts.py --device openvino
    python scripts/benchmark_tts.py --device auto
    python scripts/benchmark_tts.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project source is importable when run from the repo root.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

BENCHMARK_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow."
)

DEFAULT_VOICE = "af_sarah"
DEFAULT_LANG = "en-us"
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def _run_benchmark(
    device: str,
    text: str,
    voice: str,
    language: str,
    runs: int,
    warmup: int,
) -> dict:
    """Run synthesis benchmark for a single device. Returns metrics dict."""
    from audiobook_engine.infrastructure.tts.kokoro.backend import (
        KokoroBackend,
    )

    print(f"\n  Loading model on {device}...")
    try:
        backend = KokoroBackend(device=device)
    except Exception as exc:
        return {"device": device, "error": str(exc)}

    # Force model load
    _ = backend.capabilities()
    actual_device = backend.active_device or device

    # Get actual ORT session providers for diagnostics
    session_providers: list[str] = []
    try:
        kokoro = backend._kokoro
        if hasattr(kokoro, "sess"):
            session_providers = kokoro.sess.get_providers()
    except Exception:
        pass

    # Warmup
    print(f"  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            backend.synthesize(
                text, language, voice, Path(f.name), speed=1.0
            )

    # Benchmark
    print(f"  Benchmarking ({runs} runs)...")
    times: list[float] = []
    audio_duration = 0.0

    for _ in range(runs):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            out = Path(f.name)
            t0 = time.perf_counter()
            backend.synthesize(text, language, voice, out, speed=1.0)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            times.append(elapsed)

            # Measure audio duration from WAV header
            try:
                import wave

                with wave.open(str(out), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    audio_duration = frames / rate
            except Exception:
                pass

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    rtf = avg_time / audio_duration if audio_duration > 0 else float("inf")

    return {
        "device": device,
        "actual_device": actual_device,
        "session_providers": session_providers,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "runs": runs,
    }


def _print_result(result: dict) -> None:
    """Pretty-print a single benchmark result."""
    if "error" in result:
        print(f"\n  {result['device']}: ERROR — {result['error']}")
        return

    print(f"\n  Device:              {result['actual_device']}")
    session_providers = result.get("session_providers", [])
    if session_providers:
        print(f"  Session providers:   {session_providers}")
    print(f"  Audio duration:      {result['audio_duration']:.2f}s")
    print(f"  Avg inference:       {result['avg_time']:.3f}s")
    print(f"  Min inference:       {result['min_time']:.3f}s")
    print(f"  Max inference:       {result['max_time']:.3f}s")
    print(f"  RTF:                 {result['rtf']:.3f}x")
    print(f"  Runs:                {result['runs']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TTS devices")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "openvino"],
        help="Device to benchmark (default: auto)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all available devices",
    )
    parser.add_argument(
        "--text",
        default=BENCHMARK_TEXT,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Voice ID (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANG,
        help=f"Language code (default: {DEFAULT_LANG})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=BENCHMARK_RUNS,
        help=f"Number of benchmark runs (default: {BENCHMARK_RUNS})",
    )
    args = parser.parse_args()

    print("TTS Benchmark")
    print("=" * 40)
    print(f"Text:     {args.text[:60]}...")
    print(f"Voice:    {args.voice}")
    print(f"Language: {args.language}")

    results: list[dict] = []

    if args.all:
        from audiobook_engine.infrastructure.device import detect_devices

        devices = detect_devices()
        device_names = [d.name for d in devices if d.available]
        # Prefer openvino_gpu over openvino for benchmarking
        if "openvino_gpu" in device_names:
            device_names = ["cpu", "openvino_gpu"]
        elif "openvino" in device_names:
            device_names = ["cpu", "openvino"]
        else:
            device_names = ["cpu"]
    else:
        device_names = [args.device]

    for device in device_names:
        result = _run_benchmark(
            device, args.text, args.voice, args.language,
            args.runs, WARMUP_RUNS,
        )
        results.append(result)
        _print_result(result)

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 40)
        print("Summary")
        print("=" * 40)
        valid = [r for r in results if "error" not in r]
        if len(valid) >= 2:
            cpu_r = next((r for r in valid if r["device"] == "cpu"), None)
            gpu_r = next(
                (r for r in valid if r["device"] != "cpu"), None
            )
            if cpu_r and gpu_r:
                speedup = cpu_r["avg_time"] / gpu_r["avg_time"]
                print(f"  GPU speedup: {speedup:.2f}x")
                print(
                    f"  CPU RTF:  {cpu_r['rtf']:.3f}x"
                )
                print(
                    f"  GPU RTF:  {gpu_r['rtf']:.3f}x"
                )


if __name__ == "__main__":
    main()
