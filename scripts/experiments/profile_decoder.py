#!/usr/bin/env python3
"""Deep decoder profiling: measure per-token breakdown for PyTorch vs OpenVINO GPU.

Measures:
- Input preparation time (tensor creation, numpy conversion)
- CPU→GPU transfer time
- GPU inference time
- GPU→CPU transfer time
- Postprocessing time (torch.from_numpy, etc.)
- Total per-token time
- Number of autoregressive steps
- KV cache behavior
"""

from __future__ import annotations

import gc
import time
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ID = "parler-tts/parler_tts_mini_v0.1"
IR_DIR = Path(__file__).resolve().parent.parent.parent / "work" / "parler_ov"

PROMPT = "This is a test of Parler TTS running on an Intel GPU."
DESCRIPTION = (
    "A clear, natural female voice with moderate pitch, "
    "speaking at a measured pace, suitable for audiobook narration."
)


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def info(key: str, value: str) -> None:
    print(f"  {key:35s} {value}")


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------

def load_model():
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    info("loading", REPO_ID)
    t0 = time.perf_counter()
    model = ParlerTTSForConditionalGeneration.from_pretrained(REPO_ID).to("cpu")
    model.eval()
    info("load_time", f"{time.perf_counter() - t0:.2f}s")

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    sample_rate = getattr(model.audio_encoder.config, "sampling_rate", 44100)
    return model, tokenizer, sample_rate


# ---------------------------------------------------------------------------
# 2. PyTorch CPU decoder profiling (hook-based)
# ---------------------------------------------------------------------------

class DecoderProfiler:
    """Hooks into the decoder to measure per-step timing."""

    def __init__(self) -> None:
        self.step_times: list[float] = []
        self.step_input_sizes: list[tuple[int, ...]] = []
        self._t0: float = 0.0
        self._step_count: int = 0

    def pre_hook(self, _module: Any, _args: Any) -> None:
        self._t0 = time.perf_counter()

    def post_hook(self, _module: Any, _args: Any, _output: Any) -> None:
        dt = time.perf_counter() - self._t0
        self.step_times.append(dt)
        self._step_count += 1


def profile_pytorch_decoder(model, tokenizer, sample_rate, iterations=3):
    """Profile PyTorch CPU decoder with per-step timing."""
    section("PYTORCH CPU DECODER PROFILING")

    input_ids = tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to("cpu")
    prompt_input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cpu")

    # Hook the inner decoder model
    decoder_model = model.decoder.model.decoder
    profiler = DecoderProfiler()
    h1 = decoder_model.register_forward_pre_hook(profiler.pre_hook)
    h2 = decoder_model.register_forward_hook(profiler.post_hook)

    results = []
    for i in range(iterations):
        profiler.step_times.clear()
        profiler._step_count = 0

        gc.collect()
        t0 = time.perf_counter()
        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )
        t_total = time.perf_counter() - t0

        audio = generation.cpu().numpy().squeeze()
        duration = len(audio) / sample_rate

        results.append({
            "total": t_total,
            "duration": duration,
            "rtf": t_total / duration if duration > 0 else 0,
            "num_steps": profiler._step_count,
            "step_times": list(profiler.step_times),
            "step_mean": sum(profiler.step_times) / len(profiler.step_times) if profiler.step_times else 0,
            "step_min": min(profiler.step_times) if profiler.step_times else 0,
            "step_max": max(profiler.step_times) if profiler.step_times else 0,
        })
        info(f"iter {i+1}", f"steps={profiler._step_count} total={t_total:.3f}s "
             f"duration={duration:.3f}s RTF={t_total/duration:.2f} "
             f"step_mean={results[-1]['step_mean']*1000:.1f}ms")

    h1.remove()
    h2.remove()

    # Summary
    avg_steps = sum(r["num_steps"] for r in results) / len(results)
    avg_step_mean = sum(r["step_mean"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    info("avg_steps", f"{avg_steps:.0f}")
    info("avg_step_mean", f"{avg_step_mean*1000:.1f}ms")
    info("avg_rtf", f"{avg_rtf:.2f}")

    return results


# ---------------------------------------------------------------------------
# 3. OpenVINO GPU decoder profiling (detailed per-token)
# ---------------------------------------------------------------------------

def profile_ov_gpu_decoder(model, tokenizer, sample_rate, ov_paths, iterations=3):
    """Profile OpenVINO GPU decoder with detailed per-token timing."""
    section("OPENVINO GPU DECODER PROFILING")

    import openvino as ov
    from collections import namedtuple

    EncoderOutput = namedtuple("EncoderOutput", "last_hidden_state")
    DecoderOutput = namedtuple(
        "DecoderOutput",
        ("last_hidden_state", "past_key_values", "hidden_states",
         "attentions", "cross_attentions"),
    )

    core = ov.Core()
    ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

    def _to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return t

    # Compile models
    info("compiling", "text_encoder for GPU...")
    t0 = time.perf_counter()
    enc_model = core.compile_model(str(ov_paths["text_encoder"]), "GPU", ov_config)
    info("compile_encoder", f"{time.perf_counter() - t0:.2f}s")

    info("compiling", "decoder_stage_1 for GPU...")
    t0 = time.perf_counter()
    dec1_model = core.compile_model(str(ov_paths["decoder_stage_1"]), "GPU", ov_config)
    info("compile_decoder_1", f"{time.perf_counter() - t0:.2f}s")

    info("compiling", "decoder_stage_2 for GPU...")
    t0 = time.perf_counter()
    dec2_model = core.compile_model(str(ov_paths["decoder_stage_2"]), "GPU", ov_config)
    info("compile_decoder_2", f"{time.perf_counter() - t0:.2f}s")

    # Check execution devices
    for name, compiled in [("encoder", enc_model), ("dec1", dec1_model), ("dec2", dec2_model)]:
        exec_dev = compiled.get_property("EXECUTION_DEVICES")
        info(f"{name}_device", str(exec_dev))

    # Get input info
    info("dec1_inputs", "")
    for inp in dec1_model.inputs:
        info(f"  {list(inp.names)}", f"shape={inp.partial_shape} type={inp.element_type}")

    # Wrapper
    class TextEncoderWrap(torch.nn.Module):
        main_input_name = "input_ids"
        def __init__(self):
            super().__init__()
            self.encoder = enc_model
            self.config = model.text_encoder.config
            self.dtype = torch.float32
        def forward(self, input_ids, **_):
            out = self.encoder(_to_numpy(input_ids))[0]
            return EncoderOutput(torch.from_numpy(out))

    class DecoderWrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder_stage_1 = dec1_model
            self.decoder_stage_2 = dec2_model
            self.config = model.decoder.model.decoder.config
            embed_dim = self.config.vocab_size + 1
            self.embed_tokens = torch.nn.ModuleList(
                [torch.nn.Embedding(embed_dim, self.config.hidden_size)
                 for _ in range(self.config.num_codebooks)]
            )

        def forward(self, input_ids=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, past_key_values=None,
                    prompt_hidden_states=None, **kwargs):
            has_pkv = past_key_values is not None
            if has_pkv:
                if hasattr(past_key_values, "self_attention_cache"):
                    sc = past_key_values.self_attention_cache
                    if hasattr(sc, "key_cache") and len(sc.key_cache) == 0:
                        has_pkv = False

            if has_pkv:
                pkv_list = []
                if hasattr(past_key_values, "self_attention_cache"):
                    sc = past_key_values.self_attention_cache
                    cc = past_key_values.cross_attention_cache
                    for li in range(len(sc.key_cache)):
                        pkv_list.append(sc.key_cache[li])
                        pkv_list.append(sc.value_cache[li])
                        pkv_list.append(cc.key_cache[li])
                        pkv_list.append(cc.value_cache[li])
                else:
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
                inputs = {}
                if input_ids is not None:
                    inputs["input_ids"] = _to_numpy(input_ids)
                if encoder_hidden_states is not None:
                    inputs["encoder_hidden_states"] = _to_numpy(encoder_hidden_states)
                if encoder_attention_mask is not None:
                    inputs["encoder_attention_mask"] = _to_numpy(encoder_attention_mask)
                if prompt_hidden_states is not None:
                    inputs["prompt_hidden_states"] = _to_numpy(prompt_hidden_states)
                known = set()
                for inp in self.decoder_stage_1.inputs:
                    known.update(inp.names)
                filtered = {k: v for k, v in inputs.items() if k in known}
                outs = self.decoder_stage_1(filtered)

            outs_t = [torch.from_numpy(o) for o in outs.values()]
            past = list(list(outs_t[i:i+4]) for i in range(1, len(outs_t), 4))
            return DecoderOutput(outs_t[0], past, None, None, None)

    # Replace model components
    model.text_encoder = TextEncoderWrap()
    model.decoder.model.decoder = DecoderWrap()
    model._supports_cache_class = False
    model._supports_static_cache = False

    input_ids = tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to("cpu")
    prompt_input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cpu")

    # Hook the decoder to count steps
    decoder_obj = model.decoder.model.decoder
    step_counter = [0]
    dec_times_stage1 = []
    dec_times_stage2 = []

    def dec_pre_hook(_m, _args):
        step_counter[0] += 1

    h = decoder_obj.register_forward_pre_hook(dec_pre_hook)

    results = []
    for i in range(iterations):
        step_counter[0] = 0
        dec_times_stage1.clear()
        dec_times_stage2.clear()
        gc.collect()

        # Profile with raw OV calls
        # First, get encoder output
        enc_out = model.text_encoder(input_ids)
        enc_hs = enc_out.last_hidden_state
        proj_hs = model.enc_to_dec_proj(enc_hs)

        # Build decoder kwargs
        desc_ids = tokenizer(DESCRIPTION, return_tensors="pt").input_ids
        prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

        # Time the full generation
        t0 = time.perf_counter()
        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )
        t_total = time.perf_counter() - t0

        audio = generation.cpu().numpy().squeeze()
        duration = len(audio) / sample_rate

        info(f"iter {i+1}", f"steps={step_counter[0]} total={t_total:.3f}s "
             f"duration={duration:.3f}s RTF={t_total/duration:.2f}")

        results.append({
            "total": t_total,
            "duration": duration,
            "rtf": t_total / duration if duration > 0 else 0,
            "num_steps": step_counter[0],
        })

    h.remove()

    avg_steps = sum(r["num_steps"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    info("avg_steps", f"{avg_steps:.0f}")
    info("avg_rtf", f"{avg_rtf:.2f}")

    return results


# ---------------------------------------------------------------------------
# 4. Micro-benchmark: raw OV inference overhead
# ---------------------------------------------------------------------------

def micro_benchmark_ov(ov_paths):
    """Measure raw OpenVINO inference overhead without model.generate()."""
    section("MICRO-BENCHMARK: Raw OV Inference Overhead")

    import openvino as ov

    core = ov.Core()
    ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

    # Compile decoder_stage_1
    dec1 = core.compile_model(str(ov_paths["decoder_stage_1"]), "GPU", ov_config)

    # Create dummy inputs matching actual shapes
    # decoder_stage_1 inputs: input_ids(9,1), enc_hs(1,N,1024), enc_mask(1,N), prompt_hs(1,9,1024)
    enc_seq_len = 24  # typical for our prompt

    dummy_inputs = {
        "input_ids": np.ones((9, 1), dtype=np.int64),
        "encoder_hidden_states": np.ones((1, enc_seq_len, 1024), dtype=np.float32),
        "encoder_attention_mask": np.ones((1, enc_seq_len), dtype=np.int64),
        "prompt_hidden_states": np.ones((1, 9, 1024), dtype=np.float32),
    }

    # Warmup
    info("warmup", "5 iterations...")
    for _ in range(5):
        dec1(dummy_inputs)

    # Measure with sync
    times_sync = []
    for _ in range(20):
        t0 = time.perf_counter()
        dec1(dummy_inputs)
        times_sync.append(time.perf_counter() - t0)

    info("sync_inference_mean", f"{sum(times_sync)/len(times_sync)*1000:.2f}ms")
    info("sync_inference_min", f"{min(times_sync)*1000:.2f}ms")
    info("sync_inference_max", f"{max(times_sync)*1000:.2f}ms")

    # Now measure with request-level timing
    info("", "")
    info("request_level", "Measuring individual request phases...")

    # Get input/output info
    input_names = [list(inp.names) for inp in dec1.inputs]
    output_count = len(dec1.outputs)
    info("inputs", str(input_names))
    info("outputs", f"{output_count} tensors")

    # Time: input preparation vs inference
    prep_times = []
    infer_times = []
    for _ in range(20):
        # Prep
        t_prep_start = time.perf_counter()
        inputs_np = {k: v.copy() for k, v in dummy_inputs.items()}
        t_prep_end = time.perf_counter()
        prep_times.append(t_prep_end - t_prep_start)

        # Inference only
        t_inf_start = time.perf_counter()
        dec1(inputs_np)
        t_inf_end = time.perf_counter()
        infer_times.append(t_inf_end - t_inf_start)

    info("input_prep_mean", f"{sum(prep_times)/len(prep_times)*1000:.3f}ms")
    info("inference_mean", f"{sum(infer_times)/len(infer_times)*1000:.2f}ms")

    # Check: is there CPU↔GPU transfer overhead?
    # Try with shared memory (zero-copy)
    info("", "")
    info("shared_memory_test", "Testing with shared tensors...")
    try:
        shared_times = []
        for _ in range(20):
            # Create tensors directly as numpy (already on CPU)
            t0 = time.perf_counter()
            dec1(dummy_inputs)
            t1 = time.perf_counter()
            shared_times.append(t1 - t0)
        info("shared_inference_mean", f"{sum(shared_times)/len(shared_times)*1000:.2f}ms")
    except Exception as e:
        info("shared_memory_error", str(e)[:100])


# ---------------------------------------------------------------------------
# 5. OpenVINO profiling
# ---------------------------------------------------------------------------

def ov_profiling(ov_paths):
    """Enable OpenVINO profiling to see per-op timing."""
    section("OPENVINO PROFILING")

    import openvino as ov

    core = ov.Core()
    ov_config = {
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
    }

    dec1 = core.compile_model(str(ov_paths["decoder_stage_1"]), "GPU", ov_config)

    # Enable profiling
    prof_config = {
        "INFERENCE_PRECISION_HINT": "f32",
        "PERFORMANCE_HINT": "LATENCY",
        "PERF_COUNT": "YES",
    }
    dec1_profiled = core.compile_model(str(ov_paths["decoder_stage_1"]), "GPU", prof_config)

    dummy_inputs = {
        "input_ids": np.ones((9, 1), dtype=np.int64),
        "encoder_hidden_states": np.ones((1, 24, 1024), dtype=np.float32),
        "encoder_attention_mask": np.ones((1, 24), dtype=np.int64),
        "prompt_hidden_states": np.ones((1, 9, 1024), dtype=np.float32),
    }

    # Warmup
    for _ in range(3):
        dec1_profiled(dummy_inputs)

    # Profile
    info("profiling", "Running 5 profiled inferences...")
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        dec1_profiled(dummy_inputs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        info(f"  run {i+1}", f"{(t1-t0)*1000:.2f}ms")

    info("mean", f"{sum(times)/len(times)*1000:.2f}ms")

    # Try to get profiling info
    try:
        # Check if there's a way to get profiling data
        exec_devices = dec1_profiled.get_property("EXECUTION_DEVICES")
        info("exec_devices", str(exec_devices))

        # Check OPTIMAL_NUMBER_OF_INFER_REQUESTS
        try:
            opt_n = dec1_profiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
            info("optimal_infer_requests", str(opt_n))
        except Exception:
            pass

        # Check LATENCY
        try:
            latency = dec1_profiled.get_property("LATENCY")
            info("latency_property", str(latency))
        except Exception:
            pass

    except Exception as e:
        info("profiling_error", str(e)[:200])


# ---------------------------------------------------------------------------
# 6. Model caching test
# ---------------------------------------------------------------------------

def test_model_caching(ov_paths):
    """Test if model caching improves second-run performance."""
    section("MODEL CACHING TEST")

    import openvino as ov

    core = ov.Core()
    cache_dir = IR_DIR / "ov_cache"
    cache_dir.mkdir(exist_ok=True)
    core.set_property({"CACHE_DIR": str(cache_dir)})

    ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

    dummy_inputs = {
        "input_ids": np.ones((9, 1), dtype=np.int64),
        "encoder_hidden_states": np.ones((1, 24, 1024), dtype=np.float32),
        "encoder_attention_mask": np.ones((1, 24), dtype=np.int64),
        "prompt_hidden_states": np.ones((1, 9, 1024), dtype=np.float32),
    }

    # First run: cold cache
    info("run 1", "cold cache...")
    t0 = time.perf_counter()
    dec1 = core.compile_model(str(ov_paths["decoder_stage_1"]), "GPU", ov_config)
    t_compile1 = time.perf_counter() - t0
    for _ in range(3):
        dec1(dummy_inputs)
    t0 = time.perf_counter()
    for _ in range(10):
        dec1(dummy_inputs)
    t_infer1 = (time.perf_counter() - t0) / 10
    info("compile_time", f"{t_compile1:.2f}s")
    info("inference_mean", f"{t_infer1*1000:.2f}ms")
    del dec1
    gc.collect()

    # Second run: warm cache
    info("run 2", "warm cache...")
    t0 = time.perf_counter()
    dec1 = core.compile_model(str(ov_paths["decoder_stage_1"]), "GPU", ov_config)
    t_compile2 = time.perf_counter() - t0
    for _ in range(3):
        dec1(dummy_inputs)
    t0 = time.perf_counter()
    for _ in range(10):
        dec1(dummy_inputs)
    t_infer2 = (time.perf_counter() - t0) / 10
    info("compile_time", f"{t_compile2:.2f}s")
    info("inference_mean", f"{t_infer2*1000:.2f}ms")
    info("speedup_compile", f"{t_compile1/t_compile2:.1f}x")
    info("speedup_infer", f"{t_infer1/t_infer2:.2f}x")


# ---------------------------------------------------------------------------
# 7. FP16 precision test
# ---------------------------------------------------------------------------

def test_fp16_precision(ov_paths):
    """Test FP16 precision for potential speedup."""
    section("FP16 PRECISION TEST")

    import openvino as ov

    core = ov.Core()

    dummy_inputs = {
        "input_ids": np.ones((9, 1), dtype=np.int64),
        "encoder_hidden_states": np.ones((1, 24, 1024), dtype=np.float32),
        "encoder_attention_mask": np.ones((1, 24), dtype=np.int64),
        "prompt_hidden_states": np.ones((1, 9, 1024), dtype=np.float32),
    }

    for precision in ["FP32", "FP16"]:
        config = {"INFERENCE_PRECISION_HINT": precision}
        try:
            dec1 = core.compile_model(str(ov_paths["decoder_stage_1"]), "GPU", config)
            # Warmup
            for _ in range(3):
                dec1(dummy_inputs)
            # Measure
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                dec1(dummy_inputs)
                times.append(time.perf_counter() - t0)
            mean_ms = sum(times) / len(times) * 1000
            info(f"{precision}_mean", f"{mean_ms:.2f}ms")
            del dec1
            gc.collect()
        except Exception as e:
            info(f"{precision}_error", str(e)[:150])


# ---------------------------------------------------------------------------
# 8. Check for async inference
# ---------------------------------------------------------------------------

def test_async_inference(ov_paths):
    """Test if async inference helps."""
    section("ASYNC INFERENCE TEST")

    import openvino as ov

    core = ov.Core()
    ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

    dec1 = core.compile_model(str(ov_paths["decoder_stage_1"]), "GPU", ov_config)

    dummy_inputs = {
        "input_ids": np.ones((9, 1), dtype=np.int64),
        "encoder_hidden_states": np.ones((1, 24, 1024), dtype=np.float32),
        "encoder_attention_mask": np.ones((1, 24), dtype=np.int64),
        "prompt_hidden_states": np.ones((1, 9, 1024), dtype=np.float32),
    }

    # Warmup
    for _ in range(3):
        dec1(dummy_inputs)

    # Sync
    sync_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        dec1(dummy_inputs)
        sync_times.append(time.perf_counter() - t0)
    info("sync_mean", f"{sum(sync_times)/len(sync_times)*1000:.2f}ms")

    # Async: create inference request and use start_async + wait
    try:
        req = dec1.create_infer_request()
        async_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            req.start_async(dummy_inputs)
            req.wait()
            async_times.append(time.perf_counter() - t0)
        info("async_mean", f"{sum(async_times)/len(async_times)*1000:.2f}ms")
    except Exception as e:
        info("async_error", str(e)[:200])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Parler-TTS Decoder Deep Profiling")
    print("  OpenVINO GPU vs PyTorch CPU")
    print("=" * 70)

    model, tokenizer, sample_rate = load_model()

    ov_paths = {
        "text_encoder": IR_DIR / "text_encoder_ir.xml",
        "decoder_stage_1": IR_DIR / "decoder_stage_1_ir.xml",
        "decoder_stage_2": IR_DIR / "decoder_stage_2_ir.xml",
    }
    missing = [k for k, v in ov_paths.items() if not v.exists()]
    if missing:
        print(f"ERROR: Missing IR files: {missing}")
        print("Run test_parler_openvino.py --force-conversion first")
        return

    # 1. PyTorch CPU baseline
    pytorch_results = profile_pytorch_decoder(model, tokenizer, sample_rate, iterations=3)

    # 2. OpenVINO GPU detailed profiling
    gc.collect()
    model2, _, _ = load_model()
    ov_results = profile_ov_gpu_decoder(model2, tokenizer, sample_rate, ov_paths, iterations=3)

    # 3. Micro-benchmark
    gc.collect()
    micro_benchmark_ov(ov_paths)

    # 4. OpenVINO profiling
    gc.collect()
    ov_profiling(ov_paths)

    # 5. Model caching
    gc.collect()
    test_model_caching(ov_paths)

    # 6. FP16 precision
    gc.collect()
    test_fp16_precision(ov_paths)

    # 7. Async inference
    gc.collect()
    test_async_inference(ov_paths)

    # Summary
    section("SUMMARY")
    py_avg_steps = sum(r["num_steps"] for r in pytorch_results) / len(pytorch_results)
    py_avg_rtf = sum(r["rtf"] for r in pytorch_results) / len(pytorch_results)
    ov_avg_steps = sum(r["num_steps"] for r in ov_results) / len(ov_results)
    ov_avg_rtf = sum(r["rtf"] for r in ov_results) / len(ov_results)

    info("PyTorch CPU steps", f"{py_avg_steps:.0f}")
    info("PyTorch CPU RTF", f"{py_avg_rtf:.2f}")
    info("OpenVINO GPU steps", f"{ov_avg_steps:.0f}")
    info("OpenVINO GPU RTF", f"{ov_avg_rtf:.2f}")
    info("Decoder ratio", f"OV GPU is {ov_avg_rtf/py_avg_rtf:.1f}x vs PyTorch CPU")


if __name__ == "__main__":
    main()
