#!/usr/bin/env python3
"""
Streaming FP8/int8 -> BF16 converter for DeepSeek native checkpoints.

DeepSeek's official checkpoints (e.g. deepseek-ai/DeepSeek-V4-Flash-0731) ship as
int8/fp8(E4M3) weights with fp8(E8M0) block scales in a 1x16-column scheme.
ExLlamaV3's convert.py cannot read those dtypes, so this produces a clean BF16
copy that exllamav3 CAN quantize.

Format handled (DeepSeek native naming):
  <name>.weight  dtype I8 or F8_E4M3  -> dequantized to BF16 using <name>.scale
  <name>.scale   dtype F8_E8M0        -> consumed (block scale [n, k/16])
  all other tensors                   -> copied verbatim (incl. BF16/F32/I64)

Reconstruction is value = float(weight_bits) * scale (per 16-column block).
Low RAM: tensors are processed one at a time; output is streamed to disk in
~4 GiB shards.

Usage:
  python3 tools/fp8_to_bf16.py -i <fp8_model_dir> -o <bf16_model_dir>
  python3 tools/fp8_to_bf16.py -i <fp8_model_dir> -o <bf16_model_dir> --keep-fp8

--keep-fp8: only the I8 (expert) weights are dequantized to BF16. The
F8_E4M3 "carried" tensors (attention / shared-MLP: wkv, wq_a/b, wo_a/b,
shared_experts) are preserved as-is (fp8 weight + E8M0 scale), matching the
footprint of 0xSero/deepseek-v4-flash-0731-spark. exllamav3 v1.4.2 loads
F8_E4M3 + 2D E8M0 scale grids natively (its [128,128] grid dequant), so the
resulting directory is still a valid convert.py input. The I8 tensors must be
dequantized by this tool because exllamav3 assumes I8+scale means packed FP4
(low-nibble-first), which is wrong for DeepSeek's true int8 1x16 blocks.
"""
import argparse, json, os, struct, shutil, sys
import torch

TORCH_FROM_TAG = {
    "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32,
    "I64": torch.int64, "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu,
}
TAG_FROM_TORCH = {v: k for k, v in TORCH_FROM_TAG.items()}
TAG_FROM_TORCH[torch.bfloat16] = "BF16"
QUANT_WEIGHT_TAGS = ("I8", "F8_E4M3")
SCALE_TAG = "F8_E8M0"


def read_header(fh):
    nbytes = struct.unpack("<Q", fh.read(8))[0]
    return json.loads(fh.read(nbytes)), 8 + nbytes


def load_tensor(path, name, info):
    with open(path, "rb") as fh:
        _, base = read_header(fh)
        b, e = info["data_offsets"]
        fh.seek(base + b)
        raw = fh.read(e - b)
    dt = TORCH_FROM_TAG[info["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(tuple(info["shape"]))


def dequant_weight(w, scale, name, dtype_tag):
    """w: [n,k] int8/fp8, scale: [n/br, k/bc] E8M0 -> bf16.

    DeepSeek native block scheme (empirically verified on DS-V4-Flash-0731):
      I8      -> per (1x16)   block
      F8_E4M3 -> per (128x128) block
    """
    wf = w.float()
    s = scale.float()
    if wf.dim() != 2 or s.dim() != 2:
        raise ValueError(f"[{name}] unexpected dims weight {tuple(w.shape)} scale {tuple(s.shape)}")
    n, k = wf.shape
    nr, nc = s.shape
    if dtype_tag == "I8":
        br, bc = 1, 16
    elif dtype_tag == "F8_E4M3":
        br, bc = 128, 128
    else:
        raise ValueError(f"[{name}] unknown quant dtype {dtype_tag}")
    if nr * br != n or nc * bc != k:
        raise ValueError(f"[{name}] scale {tuple(s.shape)} incompatible with weight {tuple(w.shape)} (block {br}x{bc})")
    # expand block scale: rows then cols (no huge intermediate needed; result is [n,k] fp32)
    s = s.unsqueeze(1).expand(nr, br, nc).reshape(n, nc)
    s = s.unsqueeze(2).expand(n, nc, bc).reshape(n, k)
    return (wf * s).to(torch.bfloat16)


def tensor_bytes(t):
    """Raw little-endian bytes of a contiguous CPU tensor (any dtype incl. bfloat16)."""
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_dir", required=True)
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--shard_gb", type=float, default=4.0)
    ap.add_argument("--keep-fp8", action="store_true",
                    help="keep carried F8_E4M3 tensor pairs fp8 (weight+scale); only I8 experts -> BF16")
    args = ap.parse_args()
    keep_fp8 = args.keep_fp8

    in_dir = args.in_dir.rstrip("/")
    out_dir = args.out_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    idx_path = os.path.join(in_dir, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        sys.exit("!! model.safetensors.index.json not found (is this a sharded HF model dir?)")

    print(" -- Reading shard headers ...")
    in_shards = sorted(
        p for p in os.listdir(in_dir)
        if p.startswith("model-") and p.endswith(".safetensors"))
    if not in_shards:
        sys.exit("!! no model-*.safetensors shards found")
    tint = {}                       # name -> (dtype, shape, shard, beg, end, nelem, bytes)
    for sh in in_shards:
        path = os.path.join(in_dir, sh)
        with open(path, "rb") as fh:
            hdr, _ = read_header(fh)
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            b, e = info["data_offsets"]
            tint[name] = (info["dtype"], tuple(info["shape"]), sh, b, e, e - b)

    print(f"    tensors: {len(tint)}")

    # --- classify: weights needing dequant, scales consumed, passthrough -----------------
    passthrough = {}                 # name -> info (copied verbatim)
    dequant = {}                     # weight name -> scale name (dequantized to BF16)
    for name, (dt, shape, sh, b, e, nb) in tint.items():
        if dt == SCALE_TAG:
            continue                 # decided below
        if dt in QUANT_WEIGHT_TAGS:
            cand = name[:-7] if name.endswith(".weight") else name
            sname = cand + ".scale"
            has_scale = sname in tint and tint[sname][0] == SCALE_TAG
            if has_scale and keep_fp8 and dt == "F8_E4M3":
                # carried fp8 tensors stay fp8 (weight + scale both kept verbatim)
                passthrough[name] = tint[name]
            elif has_scale:
                dequant[name] = sname
            else:
                # quant-like tensor without scale -> lossy direct bf16 (should not happen here)
                dequant[name] = None
        else:
            passthrough[name] = tint[name]
    n_consumed_scales = len({s for s in dequant.values() if s})
    print(f"    to dequantize: {len(dequant)}  | scales consumed: {n_consumed_scales}  | kept fp8: {sum(1 for n in passthrough if n.endswith('.weight') and tint[n][0]=='F8_E4M3')}  | passthrough: {len(passthrough)}")

    # --- plan output shards ---------------------------------------------------------------
    def _numel(shape):
        ne = 1
        for d in shape:
            ne *= d
        return ne

    def out_bytes(name):
        # only true dequant targets inflate to bf16 (2B); carried fp8 / passthrough keep bytes
        return _numel(tint[name][1]) * 2 if name in dequant else tint[name][5]

    cap = int(args.shard_gb * (2**30))
    names_ordered = [n for n in tint if n not in {s for s in dequant.values() if s}]
    groups = []
    cur = []
    cur_bytes = 0
    for name in names_ordered:
        nb = out_bytes(name)
        if cur and cur_bytes + nb > cap:
            groups.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(name)
        cur_bytes += nb
    if cur:
        groups.append(cur)
    n_out = len(groups)
    print(f"    output shards: {n_out}")

    # --- write shards ----------------------------------------------------------------------
    shard_files = {}
    total_size = 0
    for shard_idx, names in enumerate(groups, 1):
        out_name = f"model-{shard_idx:05d}-of-{n_out:05d}.safetensors"
        path = os.path.join(out_dir, out_name)
        entries = []                 # (name, tensor) to serialize
        hdr = {"__metadata__": {"format": "pt"}}
        off = 0
        for name in names:
            dt, shape, sh, b, e, nb = tint[name]
            if name in dequant:
                w = load_tensor(os.path.join(in_dir, sh), name, {"dtype": dt, "shape": shape, "data_offsets": [b, e]})
                sname = dequant[name]
                if sname:
                    sdt, ssh, s_sh, sb, se, snb = tint[sname]
                    s = load_tensor(os.path.join(in_dir, s_sh), sname, {"dtype": sdt, "shape": ssh, "data_offsets": [sb, se]})
                    t = dequant_weight(w, s, name, dt)
                else:
                    t = w.to(torch.bfloat16)
                out_tag = "BF16"
            else:
                t = load_tensor(os.path.join(in_dir, sh), name, {"dtype": dt, "shape": shape, "data_offsets": [b, e]})
                out_tag = TAG_FROM_TORCH[t.dtype]
            nbytes = t.numel() * t.element_size()
            hdr[name] = {"dtype": out_tag, "shape": list(t.shape), "data_offsets": [off, off + nbytes]}
            off += nbytes
            entries.append(t)
            total_size += nbytes
        hb = json.dumps(hdr, separators=(",", ":")).encode()
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(hb)))
            fh.write(hb)
            for t in entries:
                fh.write(tensor_bytes(t))
        shard_files[out_name] = hdr
        print(f"    wrote {out_name} ({off/2**30:.2f} GiB)")

    # --- index + copy meta files -----------------------------------------------------------
    import copy
    with open(idx_path) as f:
        old_index = json.load(f)
    new_index = {"metadata": {"total_size": total_size}}
    wm = {}
    for shard_file, hdr in shard_files.items():
        for name in hdr:
            if name != "__metadata__":
                wm[name] = shard_file
    new_index["weight_map"] = wm
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)
    for fn in os.listdir(in_dir):
        if fn.endswith(".safetensors"):
            continue
        src = os.path.join(in_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
    mode = "BF16 (carried tensors kept FP8)" if keep_fp8 else "BF16"
    print(f" -- Done. {mode} model written to {out_dir}")
    print(f"    total output: {total_size/2**30:.0f} GiB in {n_out} shards")


if __name__ == "__main__":
    main()
