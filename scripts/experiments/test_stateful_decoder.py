#!/usr/bin/env python3
"""POC: MakeStateful optimization for Parler-TTS decoder with OpenVINO GPU.

Memory-efficient: loads model once, swaps wrappers for each config.
"""

from __future__ import annotations

import gc
import json
import time
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ID = "parler-tts/parler_tts_mini_v0.1"
IR_DIR = Path(__file__).resolve().parent.parent.parent / "work" / "parler_ov"
STATEFUL_DIR = IR_DIR / "stateful"
OUTPUT_DIR = IR_DIR / "output"

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


def _to_numpy(t: Any) -> Any:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


EncoderOutput = namedtuple("EncoderOutput", "last_hidden_state")
DecoderOutput = namedtuple(
    "DecoderOutput",
    ("last_hidden_state", "past_key_values", "hidden_states",
     "attentions", "cross_attentions"),
)


# ---------------------------------------------------------------------------
# 1. Create stateful decoder
# ---------------------------------------------------------------------------

def create_stateful_decoder(ov_paths, force=False):
    section("1. CREATE STATEFUL DECODER")
    import openvino as ov
    from openvino._pyopenvino._offline_transformations import (
        apply_make_stateful_transformation,
    )

    stateful_path = STATEFUL_DIR / "decoder_stateful_ir.xml"
    if stateful_path.exists() and not force:
        info("status", "exists, skipping")
        return stateful_path

    STATEFUL_DIR.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    model = core.read_model(str(ov_paths["decoder_stage_2"]))

    params = list(model.get_parameters())
    results = list(model.get_results())
    info("before", f"params={len(params)} results={len(results)}")

    kv_params = params[4:]
    kv_results = results[1:]
    pairs = list(zip(kv_params, kv_results[:len(kv_params)]))
    info("pairs", str(len(pairs)))

    t0 = time.perf_counter()
    apply_make_stateful_transformation(model, pairs)
    info("transform", f"{time.perf_counter() - t0:.3f}s")

    new_params = list(model.get_parameters())
    new_results = list(model.get_results())
    state_ops = sum(1 for op in model.get_ops()
                    if "Assign" in op.get_type_name() or "ReadValue" in op.get_type_name())
    info("after", f"params={len(new_params)} results={len(new_results)} state_ops={state_ops}")

    ov.save_model(model, str(stateful_path))
    info("saved", str(stateful_path))
    return stateful_path


# ---------------------------------------------------------------------------
# 2. Inspect
# ---------------------------------------------------------------------------

def inspect_stateful(stateful_path):
    section("2. INSPECT STATEFUL MODEL")
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(stateful_path))

    params = list(model.get_parameters())
    results = list(model.get_results())

    info("params", str(len(params)))
    info("results", str(len(results)))

    print("\n  Inputs:")
    for p in params:
        print(f"    {p.get_friendly_name()}: {p.output(0).partial_shape}")

    print("\n  Outputs:")
    for i, r in enumerate(results):
        print(f"    [{i}] {r.get_name()}")

    state_ops = []
    for op in model.get_ops():
        tn = op.get_type_name()
        if "Assign" in tn or "ReadValue" in tn:
            state_ops.append(f"{tn}: {op.get_friendly_name()}")
    print(f"\n  State ops: {len(state_ops)}")
    for s in state_ops[:5]:
        print(f"    {s}")
    if len(state_ops) > 5:
        print(f"    ... +{len(state_ops)-5} more")

    # Compare with stateless
    m2 = core.read_model(str(IR_DIR / "decoder_stage_2_ir.xml"))
    info("stateless_params", str(len(list(m2.get_parameters()))))
    info("stateless_results", str(len(list(m2.get_results()))))
    info("stateful_params", str(len(params)))
    info("stateful_results", str(len(results)))
    info("kv_inputs_eliminated", str(len(list(m2.get_parameters())) - len(params)))
    info("kv_outputs_eliminated", str(len(list(m2.get_results())) - len(results)))


# ---------------------------------------------------------------------------
# 3. Wrappers (compact)
# ---------------------------------------------------------------------------

class TextEncoderWrap(torch.nn.Module):
    main_input_name = "input_ids"
    def __init__(self, enc_model, config):
        super().__init__()
        self.encoder = enc_model
        self.config = config
        self.dtype = torch.float32
    def forward(self, input_ids, **_):
        out = self.encoder(_to_numpy(input_ids))[0]
        return EncoderOutput(torch.from_numpy(out))


class StatelessDecoderWrap(torch.nn.Module):
    def __init__(self, dec1, dec2, config):
        super().__init__()
        self.decoder_stage_1 = dec1
        self.decoder_stage_2 = dec2
        self.config = config
        embed_dim = config.vocab_size + 1
        self.embed_tokens = torch.nn.ModuleList(
            [torch.nn.Embedding(embed_dim, config.hidden_size)
             for _ in range(config.num_codebooks)])

    def forward(self, input_ids=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_values=None,
                prompt_hidden_states=None, **kwargs):
        has_pkv = past_key_values is not None
        if has_pkv and hasattr(past_key_values, "self_attention_cache"):
            sc = past_key_values.self_attention_cache
            if hasattr(sc, "key_cache") and len(sc.key_cache) == 0:
                has_pkv = False

        if has_pkv:
            pkv_list = []
            if hasattr(past_key_values, "self_attention_cache"):
                sc, cc = past_key_values.self_attention_cache, past_key_values.cross_attention_cache
                for li in range(len(sc.key_cache)):
                    pkv_list.extend([sc.key_cache[li], sc.value_cache[li],
                                     cc.key_cache[li], cc.value_cache[li]])
            else:
                for layer in past_key_values:
                    pkv_list.extend(layer if isinstance(layer, (list, tuple)) else [layer])
            arguments = (_to_numpy(input_ids), _to_numpy(encoder_hidden_states),
                         _to_numpy(encoder_attention_mask),
                         *[_to_numpy(p) for p in pkv_list])
            outs = self.decoder_stage_2(arguments)
        else:
            inputs = {}
            for name in ["input_ids", "encoder_hidden_states", "encoder_attention_mask",
                         "prompt_hidden_states"]:
                val = locals()[name]
                if val is not None:
                    inputs[name] = _to_numpy(val)
            known = set()
            for inp in self.decoder_stage_1.inputs:
                known.update(inp.names)
            outs = self.decoder_stage_1({k: v for k, v in inputs.items() if k in known})

        outs_t = [torch.from_numpy(o) for o in outs.values()]
        past = list(list(outs_t[i:i+4]) for i in range(1, len(outs_t), 4))
        return DecoderOutput(outs_t[0], past, None, None, None)


class StatefulDecoderWrap(torch.nn.Module):
    def __init__(self, dec1, dec_stateful, config):
        super().__init__()
        self.decoder_stage_1 = dec1
        self.decoder_stateful = dec_stateful
        self.config = config
        embed_dim = config.vocab_size + 1
        self.embed_tokens = torch.nn.ModuleList(
            [torch.nn.Embedding(embed_dim, config.hidden_size)
             for _ in range(config.num_codebooks)])

    def forward(self, input_ids=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_values=None,
                prompt_hidden_states=None, **kwargs):
        has_pkv = past_key_values is not None
        if has_pkv and hasattr(past_key_values, "self_attention_cache"):
            sc = past_key_values.self_attention_cache
            if hasattr(sc, "key_cache") and len(sc.key_cache) == 0:
                has_pkv = False

        if has_pkv:
            # Stateful: only main inputs, KV cache is internal state
            inputs = {
                "input_ids": _to_numpy(input_ids),
                "encoder_hidden_states": _to_numpy(encoder_hidden_states),
                "encoder_attention_mask": _to_numpy(encoder_attention_mask),
            }
            outs = self.decoder_stateful(inputs)
        else:
            # First call: stage_1 (initializes KV cache state)
            inputs = {}
            for name in ["input_ids", "encoder_hidden_states", "encoder_attention_mask",
                         "prompt_hidden_states"]:
                val = locals()[name]
                if val is not None:
                    inputs[name] = _to_numpy(val)
            known = set()
            for inp in self.decoder_stage_1.inputs:
                known.update(inp.names)
            outs = self.decoder_stage_1({k: v for k, v in inputs.items() if k in known})

        outs_t = [torch.from_numpy(o) for o in outs.values()]
        past = list(list(outs_t[i:i+4]) for i in range(1, len(outs_t), 4))
        return DecoderOutput(outs_t[0], past, None, None, None)


# ---------------------------------------------------------------------------
# 4. Benchmark (single model, swap wrappers)
# ---------------------------------------------------------------------------

class StepCounter:
    def __init__(self):
        self.count = 0
    def pre_hook(self, _m, _args):
        self.count += 1


def run_inference(model, tokenizer, sample_rate):
    """Single inference, returns timing dict."""
    input_ids = tokenizer(DESCRIPTION, return_tensors="pt").input_ids.to("cpu")
    prompt_input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cpu")

    sc = StepCounter()
    h = model.decoder.model.decoder.register_forward_pre_hook(sc.pre_hook)
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        total = time.perf_counter() - t0
        audio = gen.cpu().numpy().squeeze()
        duration = len(audio) / sample_rate
        return {"total": total, "duration": duration, "rtf": total/duration,
                "steps": sc.count, "audio": audio, "error": ""}
    except Exception as e:
        return {"total": 0, "duration": 0, "rtf": 0, "steps": 0,
                "audio": None, "error": str(e)[:300]}
    finally:
        h.remove()


def benchmark(model, tokenizer, sample_rate, label, iterations=3):
    section(f"BENCHMARK: {label}")
    # Warmup
    info("warmup", "1 iter...")
    run_inference(model, tokenizer, sample_rate)
    gc.collect()

    results = []
    for i in range(iterations):
        gc.collect()
        r = run_inference(model, tokenizer, sample_rate)
        results.append(r)
        info(f"iter {i+1}", f"steps={r['steps']} total={r['total']:.2f}s "
             f"dur={r['duration']:.2f}s RTF={r['rtf']:.2f}"
             + (f" ERR={r['error'][:60]}" if r['error'] else ""))

    valid = [r for r in results if not r["error"]]
    if not valid:
        return {"label": label, "error": results[0]["error"]}

    avg = {
        "label": label,
        "steps": sum(r["steps"] for r in valid) / len(valid),
        "total": sum(r["total"] for r in valid) / len(valid),
        "duration": sum(r["duration"] for r in valid) / len(valid),
        "rtf": sum(r["rtf"] for r in valid) / len(valid),
    }
    avg["ms_per_step"] = avg["total"] / avg["steps"] * 1000 if avg["steps"] > 0 else 0

    info("avg_steps", f"{avg['steps']:.0f}")
    info("avg_total", f"{avg['total']:.3f}s")
    info("avg_rtf", f"{avg['rtf']:.2f}")
    info("avg_ms_step", f"{avg['ms_per_step']:.1f}ms")

    # Save last audio
    last_audio = valid[-1]["audio"]
    if last_audio is not None:
        wav_path = OUTPUT_DIR / f"{label.replace(' ', '_').replace(':', '')}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(str(wav_path), last_audio, sample_rate)
        # Validate
        amp = np.abs(last_audio).max()
        has_nan = np.any(np.isnan(last_audio))
        info("audio", f"{wav_path} amp={amp:.4f} nan={has_nan}")

    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Parler-TTS MakeStateful POC")
    print("=" * 70)

    # Versions
    section("VERSIONS")
    import importlib.metadata as md
    for pkg in ["openvino", "optimum", "optimum-intel", "transformers", "torch", "parler-tts"]:
        try:
            ver = md.version(pkg.replace("-", "_"))
        except Exception:
            try: ver = md.version(pkg)
            except Exception: ver = "N/A"
        info(pkg, ver)

    # Load model ONCE
    section("LOAD MODEL")
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import openvino as ov

    info("model", REPO_ID)
    t0 = time.perf_counter()
    model = ParlerTTSForConditionalGeneration.from_pretrained(REPO_ID).to("cpu")
    model.eval()
    info("load_time", f"{time.perf_counter() - t0:.2f}s")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    sample_rate = getattr(model.audio_encoder.config, "sampling_rate", 44100)
    text_enc_cfg = model.text_encoder.config
    dec_cfg = model.decoder.model.decoder.config

    # IR paths
    ov_paths = {
        "text_encoder": IR_DIR / "text_encoder_ir.xml",
        "decoder_stage_1": IR_DIR / "decoder_stage_1_ir.xml",
        "decoder_stage_2": IR_DIR / "decoder_stage_2_ir.xml",
    }
    missing = [k for k, v in ov_paths.items() if not v.exists()]
    if missing:
        print(f"ERROR: Missing IR: {missing}"); return

    # Create stateful
    stateful_path = create_stateful_decoder(ov_paths, force=True)
    inspect_stateful(stateful_path)

    # GPU
    core = ov.Core()
    gpu_ids = [d for d in core.available_devices if d.startswith("GPU")]
    if not gpu_ids:
        print("ERROR: No GPU"); return
    ov_device = gpu_ids[0]
    info("gpu", ov_device)

    # Verify GPU
    section("GPU VERIFICATION")
    for name, path in [("enc", ov_paths["text_encoder"]),
                       ("dec1", ov_paths["decoder_stage_1"]),
                       ("dec2", ov_paths["decoder_stage_2"]),
                       ("stateful", stateful_path)]:
        try:
            c = core.compile_model(str(path), ov_device, {"INFERENCE_PRECISION_HINT": "f32"})
            info(name, f"→ {c.get_property('EXECUTION_DEVICES')}")
        except Exception as e:
            info(name, f"FAIL: {str(e)[:80]}")

    # Compile models ONCE
    section("COMPILE MODELS")
    ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

    info("compiling", "encoder...")
    t0 = time.perf_counter()
    enc_compiled = core.compile_model(str(ov_paths["text_encoder"]), ov_device, ov_config)
    info("encoder", f"{time.perf_counter()-t0:.2f}s → {enc_compiled.get_property('EXECUTION_DEVICES')}")

    info("compiling", "decoder_stage_1...")
    t0 = time.perf_counter()
    dec1_compiled = core.compile_model(str(ov_paths["decoder_stage_1"]), ov_device, ov_config)
    info("decoder_stage_1", f"{time.perf_counter()-t0:.2f}s")

    info("compiling", "decoder_stage_2...")
    t0 = time.perf_counter()
    dec2_compiled = core.compile_model(str(ov_paths["decoder_stage_2"]), ov_device, ov_config)
    info("decoder_stage_2", f"{time.perf_counter()-t0:.2f}s")

    info("compiling", "decoder_stateful...")
    t0 = time.perf_counter()
    dec_stateful_compiled = core.compile_model(str(stateful_path), ov_device, ov_config)
    info("decoder_stateful", f"{time.perf_counter()-t0:.2f}s → {dec_stateful_compiled.get_property('EXECUTION_DEVICES')}")

    # FP16 stateful
    info("compiling", "decoder_stateful FP16...")
    t0 = time.perf_counter()
    dec_stateful_fp16 = core.compile_model(str(stateful_path), ov_device,
                                            {"INFERENCE_PRECISION_HINT": "FP16"})
    info("decoder_stateful_fp16", f"{time.perf_counter()-t0:.2f}s")

    # Benchmarks (swap wrappers on same model)
    all_results = {}

    # A. PyTorch CPU
    gc.collect()
    model.text_encoder = model.text_encoder  # reset to original PyTorch
    model.decoder.model.decoder = model.decoder.model.decoder
    # Need fresh model for PyTorch (original weights)
    del model
    gc.collect()
    from parler_tts import ParlerTTSForConditionalGeneration
    model = ParlerTTSForConditionalGeneration.from_pretrained(REPO_ID).to("cpu")
    model.eval()
    all_results["A"] = benchmark(model, tokenizer, sample_rate, "A: PyTorch CPU")

    # B. OV GPU stateless
    gc.collect()
    model.text_encoder = TextEncoderWrap(enc_compiled, text_enc_cfg)
    model.decoder.model.decoder = StatelessDecoderWrap(dec1_compiled, dec2_compiled, dec_cfg)
    model._supports_cache_class = False
    model._supports_static_cache = False
    all_results["B"] = benchmark(model, tokenizer, sample_rate, "B: OV GPU stateless")

    # C. OV GPU stateful FP32
    gc.collect()
    model.text_encoder = TextEncoderWrap(enc_compiled, text_enc_cfg)
    model.decoder.model.decoder = StatefulDecoderWrap(dec1_compiled, dec_stateful_compiled, dec_cfg)
    model._supports_cache_class = False
    model._supports_static_cache = False
    all_results["C"] = benchmark(model, tokenizer, sample_rate, "C: OV GPU stateful FP32")

    # D. OV GPU stateful FP16
    gc.collect()
    model.text_encoder = TextEncoderWrap(enc_compiled, text_enc_cfg)
    model.decoder.model.decoder = StatefulDecoderWrap(dec1_compiled, dec_stateful_fp16, dec_cfg)
    model._supports_cache_class = False
    model._supports_static_cache = False
    all_results["D"] = benchmark(model, tokenizer, sample_rate, "D: OV GPU stateful FP16")

    # Final table
    section("FINAL RESULTS")
    print()
    print(f"  {'Config':<30s} {'Stateful':>8s} {'Prec':>5s} {'Steps':>6s} "
          f"{'Decoder':>10s} {'ms/step':>8s} {'RTF':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*5} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")

    for key, label, st, pr in [("A","PyTorch CPU","No","FP32"),
                                ("B","OV GPU stateless","No","FP32"),
                                ("C","OV GPU stateful","Si","FP32"),
                                ("D","OV GPU stateful+FP16","Si","FP16")]:
        r = all_results.get(key, {})
        if isinstance(r, dict) and not r.get("error"):
            print(f"  {label:<30s} {st:>8s} {pr:>5s} {r['steps']:>6.0f} "
                  f"{r['total']:>9.2f}s {r['ms_per_step']:>7.1f} {r['rtf']:>8.2f}")
        else:
            print(f"  {label:<30s} {st:>8s} {pr:>5s} {'ERROR':>6s}")

    # Improvements
    print()
    b_dec = all_results.get("B", {}).get("total", 0)
    a_dec = all_results.get("A", {}).get("total", 0)

    if b_dec > 0:
        print("  Improvement vs OV stateless (B):")
        for key, label in [("C", "stateful FP32"), ("D", "stateful FP16")]:
            r = all_results.get(key, {})
            if isinstance(r, dict) and not r.get("error"):
                d = b_dec - r["total"]
                p = d / b_dec * 100
                print(f"    {label}: {d:+.2f}s ({p:+.1f}%)")

    if a_dec > 0:
        print("\n  vs PyTorch CPU (A):")
        for key, label in [("B", "OV stateless"), ("C", "OV stateful FP32"),
                           ("D", "OV stateful FP16")]:
            r = all_results.get(key, {})
            if isinstance(r, dict) and not r.get("error"):
                d = a_dec - r["total"]
                p = d / a_dec * 100
                print(f"    {label}: {d:+.2f}s ({p:+.1f}%)")

    # Overhead
    print("\n  Overhead vs theoretical (57.87ms/step):")
    for key, label in [("A","PyTorch"), ("B","OV stateless"),
                       ("C","OV stateful"), ("D","OV stateful+FP16")]:
        r = all_results.get(key, {})
        if isinstance(r, dict) and not r.get("error"):
            expected = r["steps"] * 0.05787
            oh = r["total"] - expected
            print(f"    {label}: total={r['total']:.2f}s expected={expected:.2f}s overhead={oh:.2f}s ({oh/r['total']*100:.0f}%)")

    # Save report
    report = {k: v for k, v in all_results.items() if isinstance(v, dict)}
    rpt_path = IR_DIR / "stateful_report.json"
    rpt_path.write_text(json.dumps(report, indent=2, default=str))
    info("report", str(rpt_path))


if __name__ == "__main__":
    main()
