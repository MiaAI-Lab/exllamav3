#!/usr/bin/env python3
"""Per-module EXL3 GEMM bandwidth audit at decode shapes.

For every Linear with an EXL3 inner: time inner.forward at M in {1, 8, 16}, report
achieved GB/s (trellis+scales bytes / time). For the largest weight classes, also
dequantize once (ext.reconstruct) and time a cuBLAS fp16 matmul on the same shape
as the practical per-shape ceiling, plus a raw-read bandwidth probe for the chip.
"""
import sys, os, time, re
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from argparse import ArgumentParser
from exllamav3 import model_init
from exllamav3.ext import exllamav3_ext as ext

def time_fn(fn, warmup = 4, iters = 12):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) / iters / 1e3   # seconds

def weight_bytes(inner):
    b = inner.trellis.nbytes
    for t in (getattr(inner, "suh", None), getattr(inner, "svh", None)):
        if t is not None: b += (t.numel() * 2 if t.dtype != torch.bfloat16 else t.numel() * 2)
    return b

def main():
    parser = ArgumentParser()
    model_init.add_args(parser)
    margs = parser.parse_args([
        "-m", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m",
        "-gs", "110", "-cs", "4096",
    ])
    model, config, cache, tokenizer = model_init.init(margs, progress = True)

    # Chip bandwidth probes: read-dominated sum + pure d2d copy (r+w)
    big = torch.empty(1024 * 1024 * 1024, dtype = torch.half, device = "cuda").normal_()
    dst = torch.empty_like(big)
    t = time_fn(lambda: big.sum(), warmup = 3, iters = 8)
    bw_read = big.numel() * 2 / t / 1e9
    t = time_fn(lambda: dst.copy_(big), warmup = 3, iters = 8)
    bw_copy = big.numel() * 2 * 2 / t / 1e9
    print(f"\nchip probes: read(sum 2GiB fp16) {bw_read:.0f} GB/s | d2d copy r+w {bw_copy:.0f} GB/s")
    del big, dst
    torch.cuda.empty_cache()

    rows = []
    seen = set()
    def walk(mo):
        if id(mo) in seen: return
        seen.add(id(mo))
        inner = getattr(mo, "inner", None)
        if inner is not None and hasattr(inner, "trellis"):
            yield mo, getattr(mo, "key", "?")
        for sub in getattr(mo, "modules", []):
            yield from walk(sub)
    all_lin = list(walk(model))
    for m, key in all_lin:
        inner = m.inner
        kb = weight_bytes(inner)
        inf, outf = inner.in_features, inner.out_features
        entry = None
        for M in (1, 8, 16):
            x = torch.randn(M, inf, dtype = torch.half, device = inner.trellis.device)
            t = time_fn(lambda: inner.forward(x, {}))
            gbps = kb / t / 1e9
            if entry is None: entry = [key, inf, outf, kb, {}, {}]
            entry[4][M] = t * 1e3          # ms
            entry[5][M] = gbps
        rows.append(entry)
        # keep inner for the reference pass
        entry.append(inner)

    # cuBLAS fp16 reference for the biggest classes (by key pattern: qkv/mlp/gate...)
    def class_of(key):
        for pat, name in (("lm_head", "lm_head"), ("qkv", "attn_qkv"), ("q_proj", "attn_q"),
                          ("o_proj", "attn_o"), ("in_proj_qkvz", "gdn_qkvz"),
                          ("in_proj_z", "gdn_z"), ("in_proj_b", "gdn_b"), ("in_proj_a", "gdn_a"),
                          ("out_proj", "o_proj"), ("gate", "mlp_gate"), ("up", "mlp_up"),
                          ("down", "mlp_down")):
            if pat in key: return name
        return "other"

    refs = {}
    by_class = {}
    inners = {e[0]: e[6] for e in rows}
    for e in rows:
        c = class_of(e[0])
        by_class.setdefault(c, []).append(e)
        if c not in refs:
            refs[c] = e
    print(f"\n{'class':10s} {'n':>3s} {'GB/each':>8s} {'M8 ms':>8s} {'M8 GB/s':>8s} {'M1 GB/s':>8s} {'M16 GB/s':>9s}")
    tot_m8 = 0.0
    for c, es in sorted(by_class.items(), key = lambda kv: -sum(x[3] for x in kv[1])):
        gb = sum(x[3] for x in es) / 1e9
        m8 = sum(x[4][8] for x in es)
        tot_m8 += m8
        m8g = sum(x[3] for x in es) / (m8 / 1e3) / 1e9 if m8 else 0
        m1g = sum(x[3] for x in es) / (sum(x[4][1] for x in es) / 1e3) / 1e9
        m16g = sum(x[3] for x in es) / (sum(x[4][16] for x in es) / 1e3) / 1e9
        print(f"{c:10s} {len(es):3d} {gb:8.3f} {m8:8.2f} {m8g:8.0f} {m1g:8.0f} {m16g:9.0f}")
    print(f"{'TOTAL':10s} {'':3s} {sum(x[3] for x in rows)/1e9:8.3f} {tot_m8:8.2f}")

    print("\nslowest modules at M=8 (GB/s):")
    rows.sort(key = lambda e: e[5][8])
    for e in rows[:12]:
        print(f"  {e[0][:70]:70s} {e[1]:5d}x{e[2]:6d} {e[3]/1e6:7.1f} MB  {e[5][8]:6.0f} GB/s")

    print("\ncuBLAS fp16 reference (dequantized same shape, M=8):")
    for c, e in refs.items():
        inner = e[6]
        try:
            w = torch.empty((inner.in_features, inner.out_features), dtype = torch.half, device = inner.trellis.device)
            ext.reconstruct(w, inner.trellis, inner.K, inner.mcg, inner.mul1)
            x = torch.randn(8, inner.in_features, dtype = torch.half, device = w.device)
            t = time_fn(lambda: torch.matmul(x, w))
            gbps_cublas = w.nbytes / t / 1e9
            gbps_q_equiv = inner.trellis.nbytes / t / 1e9
            print(f"  {c:10s} {inner.in_features}x{inner.out_features}: cublas fp16 {gbps_cublas:.0f} GB/s "
                  f"(trellis-equiv {gbps_q_equiv:.0f})  vs exl3 M8 {e[5][8]:.0f} GB/s")
            del w
            torch.cuda.empty_cache()
        except Exception as ex:
            print(f"  {c:10s} reference failed: {ex}")

if __name__ == "__main__":
    main()
