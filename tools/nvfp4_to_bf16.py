#!/usr/bin/env python3
"""
NVFP4 (ModelOpt mixed-precision) -> BF16 dequantizer.

RadixArk/Qwen3.8-27B-NVFP4 stores MLP weights as packed FP4 (E2M1) with
E4M3 block scales (group 16) + a per-tensor F32 global scale, and attention
projections as FP8 (E4M3) with F32 scales. This tool dequantizes everything
to BF16 so the checkpoint can be evaluated (PPL, etc.) in exllamav3 exactly
like the other bundles — it measures the *weight-format information loss*
(W4/W8 weight-only), the apples-to-apples counterpart to EXL3's weight-only
quantization. (Runtime activation quantization of the FP8/NVFP4 serving lane
is intentionally excluded.)

Scale layout (plain [out, in/16] vs the tensor-core 128x4 "swizzle") and
nibble order are determined EMPIRICALLY in a probe step: both interpretations
are reconstructed for a few tensors and compared against the bf16 source
checkpoint; the convention with by-far lower error wins and is applied to all
tensors. Output: clean BF16 HF dir (config/tokenizer copied from the bf16
source; mtp.* draft tensors skipped — not needed for target-side evals).

Usage:
  python3 tools/nvfp4_to_bf16.py -i <nvfp4_dir> -o <bf16_dir> [--source <bf16_source_dir>] [--probe-only]
"""
import argparse, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def unpack_fp4(w_u8, low_nibble_even):
    """uint8 [out, in/2] -> fp4 float [out, in]"""
    low = (w_u8 & 0x0F).long()
    high = (w_u8 >> 4).long()
    if low_nibble_even:
        pairs = torch.stack([low, high], dim=-1)      # [out, in/2, 2]
    else:
        pairs = torch.stack([high, low], dim=-1)
    flat = pairs.reshape(pairs.shape[0], -1)          # [out, in]
    sign = torch.where(flat >= 8, -1.0, 1.0)
    mag = FP4_MAG[flat % 8]
    return sign * mag


def unswizzle_scales(scale, rows, cols):
    """swizzled flat E4M3 scales -> logical [rows, cols] (128x4 tile transpose)"""
    rows_pad = (rows + 127) // 128 * 128
    cols_pad = (cols + 3) // 4 * 4
    s = scale.reshape(-1)
    assert s.numel() == rows_pad * cols_pad, \
        f"scale numel {s.numel()} != {rows_pad}x{cols_pad}"
    v = s.reshape(rows_pad // 128, cols_pad // 4, 4, 128)
    v = v.permute(0, 3, 1, 2).contiguous()            # [r/128, 128, c/4, 4]
    return v.reshape(rows_pad, cols_pad)[:rows, :cols]


def dequant_nvfp4(w_u8, scale_e4m3, scale_2, low_nibble_even, swizzled):
    out_rows, packed_cols = w_u8.shape
    in_cols = packed_cols * 2
    w4 = unpack_fp4(w_u8, low_nibble_even)            # [out, in]
    sc = scale_e4m3.float()
    if swizzled:
        sc = unswizzle_scales(sc, out_rows, in_cols // 16)
    else:
        assert sc.shape == (out_rows, in_cols // 16), f"plain scale shape {sc.shape}"
    sc = sc.repeat_interleave(16, dim=-1)             # [out, in]
    return (w4 * sc * float(scale_2)).to(torch.bfloat16)


def mse_vs_source(rec, src_dir, src_index, name):
    if name not in src_index["weight_map"]:
        return None
    f = src_index["weight_map"][name]
    with safe_open(os.path.join(src_dir, f), framework="pt") as sf:
        src = sf.get_tensor(name).float()
    return ((rec.float() - src) ** 2).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_dir", required=True)
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--source", default="test_models/sources/Qwen3.8-27B",
                    help="bf16 source dir for convention probing")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.in_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    wm = index["weight_map"]
    src_index = {"weight_map": {}}
    src_idx_path = os.path.join(args.source, "model.safetensors.index.json")
    if os.path.exists(src_idx_path):
        with open(src_idx_path) as f:
            src_index = json.load(f)

    # classify tensors
    names = sorted(set(wm.keys()))
    quant_sidecars = set()
    nvfp4, fp8q, plain = [], [], []
    for n in names:
        if n.endswith(".weight_scale") or n.endswith(".weight_scale_2") or n.endswith(".input_scale"):
            quant_sidecars.add(n)
            continue
        if not n.endswith(".weight"):
            plain.append(n)
            continue
        base = n[:-7]
        if (base + ".weight_scale_2") in wm:
            nvfp4.append(n)
        elif (base + ".weight_scale") in wm:
            fp8q.append(n)
        else:
            plain.append(n)
    print(f"tensors: {len(nvfp4)} NVFP4, {len(fp8q)} FP8, {len(plain)} carried, "
          f"{len(quant_sidecars)} scale sidecars")

    def get(name):
        with safe_open(os.path.join(args.in_dir, wm[name]), framework="pt") as sf:
            return sf.get_tensor(name)

    # --- probe: decide nibble order + scale layout on a few NVFP4 tensors ---
    probes = [n for n in nvfp4 if "layers.10." in n][:2] + \
             [n for n in nvfp4 if "layers.40." in n][:1]
    if not probes:
        probes = nvfp4[:3]
    best, results = None, []
    for name in probes:
        w = get(name)
        sc = get(name[:-7] + ".weight_scale")
        s2 = get(name[:-7] + ".weight_scale_2")
        src_mse = mse_vs_source  # noqa
        for ln in (True, False):
            for sw in (False, True):
                try:
                    rec = dequant_nvfp4(w, sc, s2, ln, sw)
                except AssertionError as e:
                    continue
                m = mse_vs_source(rec, args.source, src_index, name)
                if m is not None:
                    results.append((m, ln, sw, name))
    if not results:
        sys.exit("!! probe produced no comparable tensors (source index missing?)")
    results.sort()
    for m, ln, sw, name in results[:4]:
        print(f"  probe {name}: nibble_low_even={ln} swizzled={sw} mse={m:.3e}")
    # majority vote on (ln, sw) among the top half (in case one tensor is odd)
    top = results[:max(3, len(results) // 2)]
    from collections import Counter
    vote = Counter((ln, sw) for _, ln, sw, _ in top).most_common(1)[0][0]
    low_nibble_even, swizzled = vote
    print(f"convention: low_nibble_even={low_nibble_even} swizzled={swizzled} "
          f"(probe mse {results[0][0]:.3e} vs next {results[1][0]:.3e})")
    if args.probe_only:
        return

    # --- convert, streaming into ~4 GiB shards ---
    os.makedirs(args.out_dir, exist_ok=True)
    shard, shard_tensors, shard_bytes, out_wm = 0, {}, 0, {}
    n_shards_est = 14

    def flush():
        nonlocal shard, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard += 1
        fn = f"model-{shard:05d}-of-{n_shards_est:05d}.safetensors"
        save_file(shard_tensors, os.path.join(args.out_dir, fn),
                  metadata={"format": "pt"})
        for k in shard_tensors:
            out_wm[k] = fn
        print(f"  wrote {fn} ({shard_bytes / 1e9:.1f} GB)", flush=True)
        shard_tensors, shard_bytes = {}, 0

    total = 0
    for name in names:
        if name in quant_sidecars:
            continue
        if name.startswith("mtp."):
            continue                              # draft layers, not needed
        t = get(name)
        if name in nvfp4:
            sc = get(name[:-7] + ".weight_scale")
            s2 = get(name[:-7] + ".weight_scale_2")
            t = dequant_nvfp4(t, sc, s2, low_nibble_even, swizzled)
        elif name in fp8q:
            sc = get(name[:-7] + ".weight_scale")
            t = (t.float() * sc.float()).to(torch.bfloat16)
        else:
            if t.dtype not in (torch.bfloat16, torch.float32, torch.int64,
                               torch.int32, torch.uint8, torch.int8):
                t = t.to(torch.bfloat16)
        b = t.numel() * t.element_size()
        shard_tensors[name] = t.contiguous()
        shard_bytes += b
        total += b
        del t
        if shard_bytes > 4e9:
            flush()
    flush()

    # fix shard count in filenames
    final = shard
    if final != n_shards_est:
        for i in range(1, final + 1):
            old = f"model-{i:05d}-of-{n_shards_est:05d}.safetensors"
            new = f"model-{i:05d}-of-{final:05d}.safetensors"
            os.rename(os.path.join(args.out_dir, old), os.path.join(args.out_dir, new))
            for k, v in out_wm.items():
                if v == old:
                    out_wm[k] = new

    with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": out_wm}, f, indent=2)

    # metadata from the bf16 source (known-loadable config, no quant sections)
    for fn in os.listdir(args.source):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = os.path.join(args.source, fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out_dir, fn))
    print(f" == wrote {args.out_dir}: {final} shards, {total / 1e9:.1f} GB bf16")


if __name__ == "__main__":
    main()
