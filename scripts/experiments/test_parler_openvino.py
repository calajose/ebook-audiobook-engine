#!/usr/bin/env python3
"""POC: Parler-TTS Mini + OpenVINO on Intel Iris Xe GPU.

Tests whether Parler-TTS can run via OpenVINO on Intel integrated GPU,
measures RTF (real-time factor) for CPU vs GPU, and identifies exactly
which components run on which device.

Usage:
    .venv/bin/python scripts/experiments/test_parler_openvino.py
    .venv/bin/python scripts/experiments/test_parler_openvino.py --skip-gpu
    .venv/bin/python scripts/experiments/test_parler_openvino.py --skip-conversion
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "parler-tts/parler_tts_mini_v0.1"
IR_DIR = Path(__file__).resolve().parent.parent.parent / "work" / "parler_ov"
OUTPUT_DIR = IR_DIR / "output"

PROMPT = "This is a test of Parler TTS running on an Intel GPU."
DESCRIPTION = (
    "A clear, natural female voice with moderate pitch, "
    "speaking at a measured pace, suitable for audiobook narration."
)

ITERATIONS = 5  # post-compilation iterations for stable timing


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@dataclass
class ComponentTiming:
    name: str
    times: list[float] = field(default_factory=list)
    device: str = ""

    @property
    def first(self) -> float:
        return self.times[0] if self.times else 0.0

    @property
    def mean_after_first(self) -> float:
        rest = self.times[1:]
        return sum(rest) / len(rest) if rest else 0.0

    @property
    def mean_all(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0


@dataclass
class BenchmarkResult:
    config_name: str
    ov_device: str | None
    encoder_timing: ComponentTiming
    decoder_timing: ComponentTiming
    dac_timing: ComponentTiming
    total_timing: ComponentTiming
    audio_duration: float = 0.0
    sample_rate: int = 0
    audio_path: str = ""
    rtf_first: float = 0.0
    rtf_mean: float = 0.0
    peak_memory_mb: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Section printer
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def info(key: str, value: str) -> None:
    print(f"  {key:30s} {value}")


def warn(msg: str) -> None:
    print(f"  WARNING: {msg}")


def err(msg: str) -> None:
    print(f"  ERROR: {msg}")


# ---------------------------------------------------------------------------
# 1. Version checks
# ---------------------------------------------------------------------------

def check_versions() -> None:
    section("1. VERSION CHECKS")
    import platform
    info("Python", platform.python_version())

    try:
        import torch
        info("PyTorch", torch.__version__)
        info("CUDA available", str(torch.cuda.is_available()))
    except ImportError:
        info("PyTorch", "NOT INSTALLED")

    try:
        import transformers
        info("Transformers", transformers.__version__)
    except ImportError:
        info("Transformers", "NOT INSTALLED")

    try:
        import openvino as ov
        info("OpenVINO", ov.__version__)
    except ImportError:
        info("OpenVINO", "NOT INSTALLED")
        sys.exit(1)

    try:
        import soundfile
        info("soundfile", soundfile.__version__)
    except ImportError:
        info("soundfile", "NOT INSTALLED")

    try:
        import numpy as np
        info("numpy", np.__version__)
    except ImportError:
        info("numpy", "NOT INSTALLED")


# ---------------------------------------------------------------------------
# 2. OpenVINO device enumeration
# ---------------------------------------------------------------------------

def enumerate_devices() -> list[str]:
    section("2. OPENVINO DEVICES")
    import openvino as ov

    core = ov.Core()
    devices = list(core.available_devices)
    info("available_devices", str(devices))

    for dev in devices:
        try:
            name = core.get_property(dev, "FULL_DEVICE_NAME")
            info(f"  {dev}", name)
        except Exception:
            info(f"  {dev}", "(could not read name)")

    gpu_ids = [d for d in devices if d.startswith("GPU")]
    if not gpu_ids:
        warn("No GPU device found. GPU benchmark will be skipped.")
    else:
        for gid in gpu_ids:
            try:
                opt_dev = core.get_property(gid, "OPTIMIZATION_CAPABILITIES")
                info(f"  {gid} optimizations", str(opt_dev))
            except Exception:
                pass

    return devices


# ---------------------------------------------------------------------------
# 3. Model loading
# ---------------------------------------------------------------------------

def load_model():
    section("3. LOAD PARLER-TTS MODEL")
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    info("model", REPO_ID)
    t0 = time.perf_counter()
    model = ParlerTTSForConditionalGeneration.from_pretrained(REPO_ID).to("cpu")
    model.eval()
    t1 = time.perf_counter()
    info("load_time", f"{t1 - t0:.2f}s")

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    # Discover sample rate from feature extractor config
    sample_rate = getattr(model.audio_encoder.config, "sampling_rate", None)
    if sample_rate is None:
        try:
            from transformers import AutoFeatureExtractor
            fe = AutoFeatureExtractor.from_pretrained(REPO_ID)
            sample_rate = fe.sampling_rate
        except Exception:
            sample_rate = 44100
    info("sample_rate", f"{sample_rate} Hz")

    # Report model structure
    info("text_encoder", type(model.text_encoder).__name__)
    info("decoder", type(model.decoder).__name__)
    info("audio_encoder", type(model.audio_encoder).__name__)

    return model, tokenizer, sample_rate


# ---------------------------------------------------------------------------
# 4. OpenVINO IR conversion
# ---------------------------------------------------------------------------

def convert_to_ov(model, force: bool = False) -> dict[str, Path]:
    section("4. CONVERT TO OPENVINO IR")
    import torch
    import openvino as ov
    from openvino import PartialShape

    paths = {
        "text_encoder": IR_DIR / "text_encoder_ir.xml",
        "decoder_stage_1": IR_DIR / "decoder_stage_1_ir.xml",
        "decoder_stage_2": IR_DIR / "decoder_stage_2_ir.xml",
    }

    all_exist = all(p.exists() for p in paths.values())
    if all_exist and not force:
        info("status", "IR files already exist, skipping conversion")
        info("text_encoder", str(paths["text_encoder"]))
        info("decoder_stage_1", str(paths["decoder_stage_1"]))
        info("decoder_stage_2", str(paths["decoder_stage_2"]))
        return paths

    def _convert(
        module, xml_path: Path, example_input: Any,
        dynamic_inputs: dict[str, PartialShape] | None = None,
    ) -> None:
        if xml_path.exists() and not force:
            info(f"  {xml_path.name}", "already exists, skipping")
            return
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            converted = ov.convert_model(
                module,
                example_input=example_input,
                input=dynamic_inputs,
            )
        ov.save_model(converted, str(xml_path))
        t1 = time.perf_counter()
        info(f"  {xml_path.name}", f"{t1 - t0:.2f}s")
        # cleanup
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = (
            torch.jit._recursive.ConcreteTypeStore()
        )
        torch.jit._state._clear_class_state()

    # --- Text Encoder ---
    # Use dynamic shape for input_ids sequence length
    info("converting", "text_encoder (T5EncoderModel)")
    example_input = {
        "input_ids": torch.ones((1, 39), dtype=torch.int64),
    }
    _convert(
        model.text_encoder,
        paths["text_encoder"],
        example_input,
        dynamic_inputs={"input_ids": PartialShape([1, -1])},
    )

    # --- Decoder Stage 1 Wrapper ---
    # Matches official notebook: no projection inside, decoder receives 1024-dim states
    # (projection is applied by model.generate() before calling decoder)
    class DecoderStage1Wrapper(torch.nn.Module):
        def __init__(self, decoder: Any) -> None:
            super().__init__()
            self.decoder = decoder

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            prompt_hidden_states: torch.Tensor | None = None,
        ) -> Any:
            return self.decoder(
                input_ids=input_ids,
                return_dict=False,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                prompt_hidden_states=prompt_hidden_states,
            )

    dec1_input = {
        "input_ids": torch.ones((9, 1), dtype=torch.int64),
        "encoder_hidden_states": torch.ones((1, 39, 1024), dtype=torch.float32),
        "encoder_attention_mask": torch.ones((1, 39), dtype=torch.int64),
        "prompt_hidden_states": torch.ones((1, 9, 1024), dtype=torch.float32),
    }
    _convert(
        DecoderStage1Wrapper(model.decoder.model.decoder),
        paths["decoder_stage_1"],
        dec1_input,
        dynamic_inputs={
            "encoder_hidden_states": PartialShape([1, -1, 1024]),
            "encoder_attention_mask": PartialShape([1, -1]),
        },
    )

    # --- Decoder Stage 2 (with past_key_values) ---
    info("converting", "decoder_stage_2 (with past_key_values)")

    class DecoderStage2Wrapper(torch.nn.Module):
        def __init__(self, decoder: Any) -> None:
            super().__init__()
            self.decoder = decoder

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
        ) -> Any:
            if past_key_values is not None:
                past_key_values = tuple(
                    tuple(past_key_values[i : i + 4])
                    for i in range(0, len(past_key_values), 4)
                )
            return self.decoder(
                input_ids=input_ids,
                return_dict=False,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
            )

    # Build dummy past_key_values: 24 layers * 4 tensors each
    pkv_dummy = tuple(
        torch.ones(1, 16, 10, 64, dtype=torch.float32)
        if i % 4 < 2
        else torch.ones(1, 16, 39, 64, dtype=torch.float32)
        for i in range(96)
    )
    dec2_input = {
        "input_ids": torch.ones((9, 1), dtype=torch.int64),
        "encoder_hidden_states": torch.ones((1, 39, 1024), dtype=torch.float32),
        "encoder_attention_mask": torch.ones((1, 39), dtype=torch.int64),
        "past_key_values": pkv_dummy,
    }
    _convert(
        DecoderStage2Wrapper(model.decoder.model.decoder),
        paths["decoder_stage_2"],
        dec2_input,
        dynamic_inputs={
            "encoder_hidden_states": PartialShape([1, -1, 1024]),
            "encoder_attention_mask": PartialShape([1, -1]),
        },
    )

    info("conversion_complete", "all 3 IR files saved")
    return paths


# ---------------------------------------------------------------------------
# 5. Inference wrappers (from official OpenVINO notebook)
# ---------------------------------------------------------------------------

def _build_wrappers(
    ov_paths: dict[str, Path], ov_device: str,
    text_encoder_config: Any, decoder_config: Any,
) -> tuple[Any, Any, Any, Any]:
    """Return (text_encoder_wrapper, decoder_wrapper, core, compile_config)."""
    import torch
    import torch.nn as nn
    import openvino as ov

    EncoderOutput = namedtuple("EncoderOutput", "last_hidden_state")
    DecoderOutput = namedtuple(
        "DecoderOutput",
        ("last_hidden_state", "past_key_values", "hidden_states",
         "attentions", "cross_attentions"),
    )

    core = ov.Core()
    ov_config: dict[str, str] = {}
    if "GPU" in ov_device:
        ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

    def _to_numpy(t: Any) -> Any:
        """Convert a torch Tensor to numpy, pass through non-tensors."""
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return t

    class TextEncoderWrapper(torch.nn.Module):
        main_input_name = "input_ids"

        def __init__(self) -> None:
            super().__init__()
            self.encoder = core.compile_model(
                str(ov_paths["text_encoder"]), ov_device, ov_config
            )
            # Must expose text_encoder config so model.generate() knows
            # hidden_size differs from decoder and applies enc_to_dec_proj
            self.config = text_encoder_config
            self.dtype = getattr(text_encoder_config, "torch_dtype", torch.float32)

        def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
            out = self.encoder(_to_numpy(input_ids))[0]
            return EncoderOutput(torch.from_numpy(out))

    class DecoderWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder_stage_1 = core.compile_model(
                str(ov_paths["decoder_stage_1"]), ov_device, ov_config
            )
            self.decoder_stage_2 = core.compile_model(
                str(ov_paths["decoder_stage_2"]), ov_device, ov_config
            )
            self.config = decoder_config
            embed_dim = decoder_config.vocab_size + 1
            self.embed_tokens = nn.ModuleList(
                [nn.Embedding(embed_dim, decoder_config.hidden_size)
                 for _ in range(decoder_config.num_codebooks)]
            )

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
            prompt_hidden_states: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> Any:
            # Handle EncoderDecoderCache (transformers >= 4.40)
            has_pkv = past_key_values is not None
            if has_pkv:
                # Check if it's an EncoderDecoderCache with empty caches
                if hasattr(past_key_values, "self_attention_cache"):
                    self_cache = past_key_values.self_attention_cache
                    if hasattr(self_cache, "key_cache") and len(self_cache.key_cache) == 0:
                        has_pkv = False

            if has_pkv:
                # Flatten EncoderDecoderCache or legacy tuple into flat list
                pkv_list: list[torch.Tensor] = []
                if hasattr(past_key_values, "self_attention_cache"):
                    # EncoderDecoderCache: [self_k, self_v, cross_k, cross_v] * layers
                    self_cache = past_key_values.self_attention_cache
                    cross_cache = past_key_values.cross_attention_cache
                    for layer_idx in range(len(self_cache.key_cache)):
                        pkv_list.append(self_cache.key_cache[layer_idx])
                        pkv_list.append(self_cache.value_cache[layer_idx])
                        pkv_list.append(cross_cache.key_cache[layer_idx])
                        pkv_list.append(cross_cache.value_cache[layer_idx])
                else:
                    # Legacy tuple format
                    for layer in past_key_values:
                        if isinstance(layer, (list, tuple)):
                            pkv_list.extend(layer)
                        else:
                            pkv_list.append(layer)

                arguments = (
                    _to_numpy(input_ids),
                    _to_numpy(encoder_hidden_states),
                    _to_numpy(encoder_attention_mask),
                    *[_to_numpy(p) for p in pkv_list],
                )
                outs = self.decoder_stage_2(arguments)
            else:
                inputs: dict[str, Any] = {}
                if input_ids is not None:
                    inputs["input_ids"] = _to_numpy(input_ids)
                if encoder_hidden_states is not None:
                    inputs["encoder_hidden_states"] = _to_numpy(encoder_hidden_states)
                if encoder_attention_mask is not None:
                    inputs["encoder_attention_mask"] = _to_numpy(encoder_attention_mask)
                if prompt_hidden_states is not None:
                    inputs["prompt_hidden_states"] = _to_numpy(prompt_hidden_states)
                # Filter to only known inputs for the compiled model
                known = set()
                for inp in self.decoder_stage_1.inputs:
                    known.update(inp.names)
                filtered = {k: v for k, v in inputs.items() if k in known}
                outs = self.decoder_stage_1(filtered)

            outs = [torch.from_numpy(o) for o in outs.values()]
            past = list(
                list(outs[i : i + 4]) for i in range(1, len(outs), 4)
            )
            return DecoderOutput(outs[0], past, None, None, None)

    return TextEncoderWrapper(), DecoderWrapper(), core, ov_config


# ---------------------------------------------------------------------------
# 6. Run single inference with per-component timing
# ---------------------------------------------------------------------------

class _DACHook:
    """Forward hook to measure DAC decode time inside model.generate()."""

    def __init__(self) -> None:
        self.dac_time: float = 0.0
        self._t0: float = 0.0

    def pre_hook(self, _module: Any, _args: Any) -> None:
        self._t0 = time.perf_counter()

    def post_hook(self, _module: Any, _args: Any, _output: Any) -> None:
        self.dac_time += time.perf_counter() - self._t0


def run_inference(
    model: Any,
    tokenizer: Any,
    sample_rate: int,
    ov_paths: dict[str, Path] | None,
    ov_device: str | None,
    config: Any,
    prompt: str,
    description: str,
    output_path: Path,
    warmup: bool = False,
) -> BenchmarkResult:
    """Run a single inference and return timing results.

    Uses forward hooks on audio_encoder.decode to measure DAC time
    inside model.generate().  The text encoder is also timed separately.

    If ov_paths is None, runs in pure PyTorch mode.
    If ov_device is given, wraps text_encoder and decoder with OpenVINO.
    The audio encoder (DAC) always runs in PyTorch.
    """
    import torch
    import numpy as np

    config_name = (
        f"OpenVINO {ov_device}" if ov_device else "PyTorch CPU"
    )

    result = BenchmarkResult(
        config_name=config_name,
        ov_device=ov_device or "none",
        encoder_timing=ComponentTiming("text_encoder"),
        decoder_timing=ComponentTiming("decoder"),
        dac_timing=ComponentTiming("audio_encoder (DAC)"),
        total_timing=ComponentTiming("total"),
        sample_rate=sample_rate,
    )

    # --- Apply OpenVINO wrappers if requested ---
    if ov_paths is not None and ov_device is not None:
        enc_wrap, dec_wrap, core, _ = _build_wrappers(
            ov_paths, ov_device,
            text_encoder_config=model.text_encoder.config,
            decoder_config=config,
        )
        model.text_encoder = enc_wrap
        model.decoder.model.decoder = dec_wrap
        model._supports_cache_class = False
        model._supports_static_cache = False
        device_label = ov_device
    else:
        device_label = "cpu"

    # --- Tokenize ---
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to("cpu")
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

    # --- Hook DAC to measure its time inside generate() ---
    dac_hook = _DACHook()
    dac_handle_pre = model.audio_encoder.register_forward_pre_hook(
        dac_hook.pre_hook
    )
    dac_handle_post = model.audio_encoder.register_forward_hook(
        dac_hook.post_hook
    )

    try:
        # --- Measure text encoder separately (before generate) ---
        t_enc_start = time.perf_counter()
        with torch.no_grad():
            _enc_out = model.text_encoder(input_ids)
        t_enc_end = time.perf_counter()
        result.encoder_timing.times.append(t_enc_end - t_enc_start)
        result.encoder_timing.device = device_label

        # --- Full generation: includes encoder + decoder + DAC ---
        t_gen_start = time.perf_counter()
        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )
        t_gen_end = time.perf_counter()
        generate_time = t_gen_end - t_gen_start

        # generation is a Tensor of shape [1, num_samples] (audio waveform)
        # because model.generate() already decodes via DAC internally.
        audio_arr = generation.cpu().numpy().squeeze()
        duration = len(audio_arr) / sample_rate if sample_rate > 0 else 0.0

        # --- Component breakdown ---
        # generate_time includes: text_encoder + decoder + DAC (all inside)
        # We measured text_encoder separately above.
        # DAC time was captured by the hook.
        dac_time = dac_hook.dac_time

        # Decoder time = generate_time - text_encoder_time - DAC_time
        # (text_encoder is called again inside generate, but the hook
        # on audio_encoder captures only the DAC portion)
        decoder_time = generate_time - (t_enc_end - t_enc_start) - dac_time
        if decoder_time < 0:
            # Fallback: if timing overlap, just use generate - DAC
            decoder_time = generate_time - dac_time

        result.decoder_timing.times.append(max(decoder_time, 0.0))
        result.decoder_timing.device = device_label
        result.dac_timing.times.append(dac_time)
        result.dac_timing.device = "cpu"  # DAC always PyTorch CPU

        # Total measured wall clock
        totalMeasured = (t_enc_end - t_enc_start) + generate_time
        result.total_timing.times.append(totalMeasured)

        result.audio_duration = duration

        if duration > 0:
            result.rtf_first = totalMeasured / duration

        # --- Validate ---
        if np.any(np.isnan(audio_arr)) or np.any(np.isinf(audio_arr)):
            result.error = "NaN or Inf detected in audio"
        if len(audio_arr) == 0:
            result.error = "Empty audio output"
        if duration < 0.1:
            result.error = f"Audio too short: {duration:.3f}s"

        # --- Save WAV ---
        if not warmup:
            import soundfile as sf
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio_arr, sample_rate)
            result.audio_path = str(output_path)

    finally:
        dac_handle_pre.remove()
        dac_handle_post.remove()

    return result


# ---------------------------------------------------------------------------
# 7. Benchmark loop
# ---------------------------------------------------------------------------

def benchmark_config(
    model: Any,
    tokenizer: Any,
    sample_rate: int,
    ov_paths: dict[str, Path] | None,
    ov_device: str | None,
    config: Any,
    label: str,
    output_subdir: str,
    iterations: int = ITERATIONS,
) -> BenchmarkResult:
    """Run multiple iterations and aggregate timing."""
    section(f"BENCHMARK: {label}")

    # Collect peak memory
    try:
        import psutil
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        proc = None
        mem_before = 0.0

    # Warmup iteration (not counted)
    info("warmup", "running 1 warmup iteration...")
    warmup_out = OUTPUT_DIR / output_subdir / "warmup.wav"
    run_inference(
        model, tokenizer, sample_rate, ov_paths, ov_device, config,
        PROMPT, DESCRIPTION, warmup_out, warmup=True,
    )
    gc.collect()

    # Measured iterations
    agg = BenchmarkResult(
        config_name=label,
        ov_device=ov_device or "none",
        encoder_timing=ComponentTiming("text_encoder"),
        decoder_timing=ComponentTiming("decoder"),
        dac_timing=ComponentTiming("audio_encoder (DAC)"),
        total_timing=ComponentTiming("total"),
        sample_rate=sample_rate,
    )

    for i in range(iterations):
        info("iteration", f"{i + 1}/{iterations}")
        out_path = OUTPUT_DIR / output_subdir / f"test_{i}.wav"
        res = run_inference(
            model, tokenizer, sample_rate, ov_paths, ov_device, config,
            PROMPT, DESCRIPTION, out_path,
        )
        if res.error:
            agg.error = res.error
            break

        agg.encoder_timing.times.extend(res.encoder_timing.times)
        agg.decoder_timing.times.extend(res.decoder_timing.times)
        agg.dac_timing.times.extend(res.dac_timing.times)
        agg.total_timing.times.extend(res.total_timing.times)
        agg.audio_duration = res.audio_duration
        agg.rtf_first = res.rtf_first

    # Save last output as canonical
    if not agg.error:
        last_out = OUTPUT_DIR / output_subdir / "final.wav"
        import soundfile as sf
        # Re-read the last saved file
        last_file = OUTPUT_DIR / output_subdir / f"test_{iterations - 1}.wav"
        if last_file.exists():
            import shutil
            shutil.copy2(str(last_file), str(last_out))
            agg.audio_path = str(last_out)

    # Memory
    if proc is not None:
        mem_after = proc.memory_info().rss / (1024 * 1024)
        agg.peak_memory_mb = max(mem_after, mem_before)

    # RTF
    if agg.audio_duration > 0:
        agg.rtf_mean = agg.total_timing.mean_all / agg.audio_duration

    # Report
    info("audio_duration", f"{agg.audio_duration:.3f}s")
    info("sample_rate", f"{agg.sample_rate} Hz")
    info("encoder_first", f"{agg.encoder_timing.first:.4f}s")
    info("encoder_mean", f"{agg.encoder_timing.mean_after_first:.4f}s")
    info("decoder_first", f"{agg.decoder_timing.first:.4f}s")
    info("decoder_mean", f"{agg.decoder_timing.mean_after_first:.4f}s")
    info("dac_first", f"{agg.dac_timing.first:.4f}s")
    info("dac_mean", f"{agg.dac_timing.mean_after_first:.4f}s")
    info("total_first", f"{agg.total_timing.first:.4f}s")
    info("total_mean", f"{agg.total_timing.mean_after_first:.4f}s")
    info("rtf_first", f"{agg.rtf_first:.4f}")
    info("rtf_mean", f"{agg.rtf_mean:.4f}")
    if agg.peak_memory_mb > 0:
        info("peak_memory", f"{agg.peak_memory_mb:.1f} MB")
    if agg.error:
        info("ERROR", agg.error)

    return agg


# ---------------------------------------------------------------------------
# 8. GPU fallback detection
# ---------------------------------------------------------------------------

def verify_gpu_fallback(
    ov_paths: dict[str, Path],
    ov_device: str,
    config: Any,
) -> dict[str, str]:
    """Compile models on GPU and check which device actually executed."""
    section("GPU FALLBACK VERIFICATION")
    import openvino as ov

    core = ov.Core()
    results: dict[str, str] = {}

    gpu_config = {"INFERENCE_PRECISION_HINT": "f32"}

    for name, xml_path in ov_paths.items():
        try:
            compiled = core.compile_model(
                str(xml_path), ov_device, gpu_config
            )
            # Get the actual device from the compiled model
            actual = compiled.get_property("EXECUTION_DEVICES")
            results[name] = str(actual)
            info(name, f"compiled → actual devices: {actual}")
        except Exception as exc:
            results[name] = f"COMPILE_FAILED: {exc}"
            err(name, str(exc))

    return results


# ---------------------------------------------------------------------------
# 9. Audio quality validation
# ---------------------------------------------------------------------------

def validate_audio(audio_path: str, label: str) -> None:
    """Check WAV file for basic quality metrics."""
    import numpy as np
    import soundfile as sf

    info(f"--- {label} ---", "")
    try:
        data, sr = sf.read(audio_path)
        info("file", audio_path)
        info("sample_rate", f"{sr} Hz")
        info("samples", f"{len(data)}")
        info("duration", f"{len(data) / sr:.3f}s")
        info("dtype", str(data.dtype))
        info("min", f"{data.min():.6f}")
        info("max", f"{data.max():.6f}")
        info("rms", f"{np.sqrt(np.mean(data ** 2)):.6f}")
        info("peak_db", f"{20 * np.log10(max(abs(data.max()), 1e-10)):.1f}")
        info("has_nan", str(np.any(np.isnan(data))))
        info("has_inf", str(np.any(np.isinf(data))))

        # Check for silence
        if np.abs(data).max() < 1e-6:
            warn("Audio is essentially silent!")
    except Exception as exc:
        err(f"Could not read WAV: {exc}")


# ---------------------------------------------------------------------------
# 10. Final report
# ---------------------------------------------------------------------------

def print_report(results: dict[str, BenchmarkResult]) -> None:
    section("FINAL REPORT")

    # Component timing table
    print("\n  Component timing breakdown (mean after warmup):")
    print(f"  {'Component':<30s} {'PyTorch CPU':>14s} {'OV CPU':>14s} {'OV GPU':>14s}")
    print(f"  {'-' * 30} {'-' * 14} {'-' * 14} {'-' * 14}")

    pytorch_res = results.get("pytorch_cpu")
    ov_cpu_res = results.get("openvino_cpu")
    ov_gpu_res = results.get("openvino_gpu")

    for comp_name, comp_key in [
        ("Text encoder", "encoder_timing"),
        ("Decoder (+ generation)", "decoder_timing"),
        ("Audio codec (DAC)", "dac_timing"),
        ("TOTAL", "total_timing"),
    ]:
        py_val = getattr(pytorch_res, comp_key).mean_after_first if pytorch_res else None
        cpu_val = getattr(ov_cpu_res, comp_key).mean_after_first if ov_cpu_res else None
        gpu_val = getattr(ov_gpu_res, comp_key).mean_after_first if ov_gpu_res else None

        py_s = f"{py_val:.4f}s" if py_val is not None else "N/A"
        cpu_s = f"{cpu_val:.4f}s" if cpu_val is not None else "N/A"
        gpu_s = f"{gpu_val:.4f}s" if gpu_val is not None else "N/A"
        print(f"  {comp_name:<30s} {py_s:>14s} {cpu_s:>14s} {gpu_s:>14s}")

    # RTF table
    print("\n  RTF (Real-Time Factor) — lower is better:")
    print(f"  {'Configuration':<30s} {'RTF (first)':>14s} {'RTF (mean)':>14s} {'Duration':>10s}")
    print(f"  {'-' * 30} {'-' * 14} {'-' * 14} {'-' * 10}")

    for key, label in [
        ("pytorch_cpu", "PyTorch CPU"),
        ("openvino_cpu", "OpenVINO CPU"),
        ("openvino_gpu", "OpenVINO GPU"),
    ]:
        res = results.get(key)
        if res and not res.error:
            print(
                f"  {label:<30s} {res.rtf_first:>14.4f} "
                f"{res.rtf_mean:>14.4f} {res.audio_duration:>9.3f}s"
            )
        elif res and res.error:
            print(f"  {label:<30s} {'ERROR':>14s} {res.error[:14]:>14s}")
        else:
            print(f"  {label:<30s} {'N/A':>14s}")

    # Classification
    section("CLASSIFICATION")
    if ov_gpu_res and not ov_gpu_res.error:
        if ov_gpu_res.rtf_mean > 0 and ov_gpu_res.rtf_mean < 2.0:
            cls = "A) FUNCIONA"
            desc = (
                "Parler-TTS + OpenVINO + Intel Iris Xe funciona correctamente. "
                "RTF < 2.0 indica rendimiento usable."
            )
        else:
            cls = "B) FUNCIONA CON LIMITACIONES"
            desc = (
                "Parler-TTS + OpenVINO + Intel GPU genera audio válido pero "
                f"RTF={ov_gpu_res.rtf_mean:.2f} indica limitaciones de rendimiento."
            )
    elif ov_gpu_res and ov_gpu_res.error:
        if "compile" in ov_gpu_res.error.lower() or "COMPILE_FAILED" in ov_gpu_res.error:
            cls = "D) NO VIABLE"
            desc = "No se pudieron compilar los modelos en GPU."
        else:
            cls = "C) CONVERSIÓN POSIBLE PERO INFERENCIA FALLIDA"
            desc = f"IR generados pero inferencia falló: {ov_gpu_res.error}"
    else:
        cls = "D) NO VIABLE"
        desc = "No se ejecutó benchmark en GPU."

    info("classification", cls)
    info("description", desc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="POC: Parler-TTS + OpenVINO on Intel Iris Xe"
    )
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip GPU benchmark (useful if no Intel GPU)",
    )
    parser.add_argument(
        "--skip-conversion", action="store_true",
        help="Skip IR conversion (use existing IR files)",
    )
    parser.add_argument(
        "--force-conversion", action="store_true",
        help="Force re-conversion even if IR files exist",
    )
    parser.add_argument(
        "--iterations", type=int, default=ITERATIONS,
        help=f"Number of measured iterations (default: {ITERATIONS})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Parler-TTS + OpenVINO POC")
    print("  Intel Iris Xe GPU feasibility test")
    print("=" * 60)

    # --- Phase 1: versions ---
    check_versions()

    # --- Phase 2: devices ---
    devices = enumerate_devices()
    gpu_ids = [d for d in devices if d.startswith("GPU")]
    has_gpu = len(gpu_ids) > 0 and not args.skip_gpu

    # --- Phase 3: load model ---
    model, tokenizer, sample_rate = load_model()
    config = model.decoder.config

    # --- Phase 4: convert ---
    if args.skip_conversion:
        section("4. CONVERT TO OPENVINO IR")
        info("status", "skipped (--skip-conversion)")
        ov_paths = {
            "text_encoder": IR_DIR / "text_encoder_ir.xml",
            "decoder_stage_1": IR_DIR / "decoder_stage_1_ir.xml",
            "decoder_stage_2": IR_DIR / "decoder_stage_2_ir.xml",
        }
        missing = [k for k, v in ov_paths.items() if not v.exists()]
        if missing:
            err("IR files missing", str(missing))
            sys.exit(1)
    else:
        ov_paths = convert_to_ov(model, force=args.force_conversion)

    # --- Phase 5: benchmark ---
    results: dict[str, BenchmarkResult] = {}

    # A) PyTorch CPU baseline
    pytorch_res = benchmark_config(
        model, tokenizer, sample_rate,
        ov_paths=None, ov_device=None, config=config,
        label="PyTorch CPU", output_subdir="pytorch_cpu",
        iterations=args.iterations,
    )
    results["pytorch_cpu"] = pytorch_res

    # B) OpenVINO GPU (primary goal)
    if has_gpu and gpu_ids:
        fallback_info = verify_gpu_fallback(ov_paths, gpu_ids[0], config)

        gc.collect()
        model_gpu, _, _ = load_model()

        ov_gpu_res = benchmark_config(
            model_gpu, tokenizer, sample_rate,
            ov_paths=ov_paths, ov_device=gpu_ids[0], config=config,
            label=f"OpenVINO {gpu_ids[0]}", output_subdir="openvino_gpu",
            iterations=args.iterations,
        )
        results["openvino_gpu"] = ov_gpu_res
    else:
        section("GPU BENCHMARK")
        info("status", "skipped (no GPU or --skip-gpu)")

    # C) OpenVINO CPU (secondary — may fail with oneDNN issues)
    gc.collect()
    model_cpu2, _, _ = load_model()

    try:
        ov_cpu_res = benchmark_config(
            model_cpu2, tokenizer, sample_rate,
            ov_paths=ov_paths, ov_device="CPU", config=config,
            label="OpenVINO CPU", output_subdir="openvino_cpu",
            iterations=args.iterations,
        )
        results["openvino_cpu"] = ov_cpu_res
    except Exception as exc:
        section("OpenVINO CPU")
        err("FAILED", str(exc)[:200])
        results["openvino_cpu"] = BenchmarkResult(
            config_name="OpenVINO CPU",
            ov_device="CPU",
            encoder_timing=ComponentTiming("text_encoder"),
            decoder_timing=ComponentTiming("decoder"),
            dac_timing=ComponentTiming("audio_encoder (DAC)"),
            total_timing=ComponentTiming("total"),
            error=f"Compilation/inference failed: {str(exc)[:300]}",
        )

    # --- Phase 6: validate audio ---
    section("6. AUDIO VALIDATION")
    for key, label in [
        ("pytorch_cpu", "PyTorch CPU"),
        ("openvino_cpu", "OpenVINO CPU"),
        ("openvino_gpu", "OpenVINO GPU"),
    ]:
        res = results.get(key)
        if res and res.audio_path:
            validate_audio(res.audio_path, label)

    # --- Phase 7: report ---
    print_report(results)

    # --- Save JSON ---
    report_path = IR_DIR / "report.json"
    report_data = {}
    for key, res in results.items():
        report_data[key] = {
            "config": res.config_name,
            "ov_device": res.ov_device,
            "audio_duration": res.audio_duration,
            "sample_rate": res.sample_rate,
            "rtf_first": res.rtf_first,
            "rtf_mean": res.rtf_mean,
            "encoder_first": res.encoder_timing.first,
            "encoder_mean": res.encoder_timing.mean_after_first,
            "decoder_first": res.decoder_timing.first,
            "decoder_mean": res.decoder_timing.mean_after_first,
            "dac_first": res.dac_timing.first,
            "dac_mean": res.dac_timing.mean_after_first,
            "total_first": res.total_timing.first,
            "total_mean": res.total_timing.mean_after_first,
            "peak_memory_mb": res.peak_memory_mb,
            "error": res.error,
            "audio_path": res.audio_path,
        }
    report_path.write_text(json.dumps(report_data, indent=2))
    info("report_saved", str(report_path))

    print(f"\n{'=' * 60}")
    print("  POC COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
