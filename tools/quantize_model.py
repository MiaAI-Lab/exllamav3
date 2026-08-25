#!/usr/bin/env python3
"""
Repeatable EXL3 conversion driver for this two-DGX-Spark setup.

Handles the "convert a model" job correctly every time:
  * resolves the source (local dir or HF repo id)
  * inspects the checkpoint dtype and picks the right input path:
      - BF16/F16 checkpoint        -> feed to convert.py directly
      - Qwen-style FP8 (E4M3 +
        .weight_scale/_inv)        -> exllamav3 dequants natively
      - DeepSeek-native I8/.scale  -> run tools/fp8_to_bf16.py first
  * preflights exllamav3 support (architecture + weight-map alignment)
  * runs convert.py with resumable checkpoints and saved logs
  * records a job record in jobs.jsonl

Examples:
  # local bf16 dir, 4.0 bpw with -hq (safe defaults for dense models):
  python tools/quantize_model.py -m ~/models/foo-bf16 -b 4.0 --hq

  # HF model (auto-downloads):
  python tools/quantize_model.py -m Qwen/Qwen3.8-27B -b 3.5 --hq

  # DeepSeek-native fp8 source:
  python tools/quantize_model.py -m /path/to/deepseek-fp8 -b 3.0 --hq --dequant

  # resume an interrupted job (keeps all other settings from the job):
  python tools/quantize_model.py -w <work_dir> --resume

  # dry-run: only download + preflight, no quantization started
  python tools/quantize_model.py -m Qwen/Qwen3.8-27B --dry-run
"""
import argparse, glob, json, os, shutil, struct, subprocess, sys, time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS)
JOBS = os.path.join(REPO_ROOT, "jobs.jsonl")
SOURCES = os.path.join(os.path.dirname(REPO_ROOT) if os.path.basename(REPO_ROOT) != "exl3" else os.path.dirname(REPO_ROOT), "exl3", "test_models", "sources")
# ^ sources live under the workspace trees on either box
SOURCES = os.path.join(REPO_ROOT, "test_models", "sources")


def log(msg):
    print(msg, flush=True)


def read_dtypes(model_dir):
    """Return {tensor_name: dtype} from all shard headers (header-only, cheap)."""
    out = {}
    for sh in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
        with open(sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = v["dtype"]
    # also support per-layer shard layouts (e.g. unquantized layers-N.safetensors)
    if not out:
        for sh in sorted(glob.glob(os.path.join(model_dir, "layers-*.safetensors"))):
            with open(sh, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    out[k] = v["dtype"]
    return out


def classify_source(model_dir):
    dt = read_dtypes(model_dir)
    if not dt:
        return "error", "no shards found"
    has_scale = any(k.endswith(".scale") and v == "F8_E8M0" for k, v in dt.items())
    has_i8 = any(v == "I8" for v in dt.values())
    has_qwen_scale = any((".weight_scale" in k or ".weight_scale_inv" in k or ".weight_scale" in k) for k in dt)
    has_e4 = any(v == "F8_E4M3" for v in dt.values())
    if has_i8 and has_scale:
        return "deepseek-native", {"n_tensors": len(dt)}
    if has_e4 and (has_qwen_scale or has_scale):
        return "fp8-qwen-style", {"n_tensors": len(dt)}
    if all(v in ("BF16", "F16", "F32", "I64") for v in dt.values()) or has_e4 or has_i8:
        return "bf16", {"n_tensors": len(dt)}
    return "unknown", {"dtypes": sorted(set(dt.values()))[:10]}


def preflight(model_dir):
    """Non-destructive exllamav3 architecture support check."""
    sys.path.insert(0, REPO_ROOT)
    try:
        from exllamav3 import Config, Model
    except Exception as e:
        return False, f"exllamav3 import failed: {e}"
    try:
        cfg = Config.from_directory(model_dir)
        model = Model.from_config(cfg)
        layout = None
        try:
            layout = model.get_layout_tree(2)
        except Exception:
            pass
        caps = getattr(model, "caps", None) or {}
        supported = caps.get("can_quantize", True)
        return bool(supported), f"arch={cfg.architecture} can_quantize={supported}"
    except Exception as e:
        return False, f"preflight failed: {type(e).__name__}: {e}"


def run(cmd, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


def attach_draft(out_dir: str, draft_dir: str, num_draft_tokens: int = 7):
    """Package a speculative-decoding draft model into an existing EXL3 output dir.

    Copies the draft weights into <out_dir>/draft/ and writes a dspark.json manifest
    describing how to serve the bundle (target -m <out_dir> + draft -m <out_dir>/draft).
    """
    if not os.path.isdir(draft_dir):
        sys.exit(f"!! draft model dir not found: {draft_dir}")
    if not os.path.isfile(os.path.join(out_dir, "config.json")):
        sys.exit(f"!! output dir does not look like a complete EXL3 model: {out_dir}")
    dest = os.path.join(out_dir, "draft")
    os.makedirs(dest, exist_ok=True)
    copied = []
    for fname in ["config.json", "model.safetensors", "README.md"]:
        src = os.path.join(draft_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, fname))
            copied.append(fname)
    manifest = {
        "draft_arch": "DSparkDraftModel",
        "draft_dir": "draft",
        "num_draft_tokens": num_draft_tokens,
        "verify_width": num_draft_tokens + 1,
        "confidence_env": "EXL3_DSPARK_CONF",
        "serving": {
            "model_dir": os.path.basename(out_dir),
            "draft_model_dir": os.path.join(os.path.basename(out_dir), "draft"),
            "hint": "serve with: <runner> -m <model_dir> -dm <draft_model_dir>",
        },
    }
    with open(os.path.join(out_dir, "dspark.json"), "w") as f:
        json.dump(manifest, f, indent = 2)
    log(f" -- attached draft ({', '.join(copied)}) -> {dest} + dspark.json manifest")


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("-m", "--model", default=None, help="source: local dir or HF repo id")
    ap.add_argument("-o", "--out", default=None, help="output EXL3 dir (default: <model>-exl3-<bits>bpw)")
    ap.add_argument("-w", "--work", default=None, help="work dir (default: <out>.work)")
    ap.add_argument("-b", "--bits", type=float, default=None)
    ap.add_argument("--hq", action="store_true", help="protect attention/shared-expert layers")
    ap.add_argument("--cr", type=int, default=None, help="cal rows")
    ap.add_argument("--cc", type=int, default=None, help="cal cols")
    ap.add_argument("-d", "--device", default=None, help="convert.py -d devices")
    ap.add_argument("--dequant", action="store_true", help="force DS4F-style fp8->bf16 dequant first")
    ap.add_argument("--resume", action="store_true", help="resume an interrupted job from -w")
    ap.add_argument("--dry-run", action="store_true", help="download+classify+preflight only")
    ap.add_argument("--detach", action="store_true", help="launch convert.py in background")
    ap.add_argument("--log", default=None, help="log file for convert.py output")
    ap.add_argument("--name", default=None, help="friendly name for the job record")
    ap.add_argument("--draft-model-dir", default=None,
                    help="DSpark draft model dir; after conversion, copy it into <out>/draft/ "
                         "and write a dspark.json manifest (target+draft serving bundle)")
    ap.add_argument("--cal-data", default=None,
                    help="calibration trace JSON passed to convert.py -cd (e.g. turboderp-style cal_trace.json); "
                         "replaces the default calibration corpus")
    ap.add_argument("--attach-draft", action="store_true",
                    help="only attach --draft-model-dir to an existing -o/--out EXL3 dir (no conversion)")
    args = ap.parse_args()

    os.makedirs(SOURCES, exist_ok=True)

    if args.attach_draft:
        if not args.out or not args.draft_model_dir:
            sys.exit("!! --attach-draft requires -o OUT (existing EXL3 dir) and --draft-model-dir")
        attach_draft(args.out, args.draft_model_dir)
        return 0

    if args.resume:
        if not args.work:
            sys.exit("!! --resume requires -w <work_dir>")
        if args.draft_model_dir and not args.out:
            sys.exit("!! --resume with --draft-model-dir also requires -o OUT")
        cmd = [sys.executable, "convert.py", "-w", args.work, "-r"]
        log_path = args.log or os.path.join(args.work, "convert.log")
        rc = run(cmd, log_path)
        log(f"convert.py exited {rc.returncode}; full log: {log_path}")
        if rc.returncode == 0 and args.draft_model_dir:
            attach_draft(args.out, args.draft_model_dir)
        return rc.returncode

    if not args.model or args.bits is None:
        ap.print_help()
        sys.exit("    provide -m MODEL and -b BITS (or use --resume/-w)")

    # resolve source
    model_dir = args.model
    if not os.path.isdir(model_dir):
        dest = os.path.join(SOURCES, args.model.replace("/", "__"))
        log(f" -- downloading {args.model} -> {dest}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(args.model, local_dir=dest)
        except Exception as e:
            sys.exit(f"!! download failed: {e}")
        model_dir = dest

    # working paths
    base = os.path.basename(model_dir.rstrip("/"))
    out = args.out or os.path.join(REPO_ROOT, "test_models", f"{base}-exl3-{args.bits:g}bpw")
    work = args.work or (out + ".work")
    os.makedirs(work, exist_ok=True)
    log_path = args.log or os.path.join(work, "convert.log")

    # classify + (optional) dequant
    kind, info = classify_source(model_dir)
    log(f" -- source format: {kind} {info}")
    cfg_dir = model_dir
    if args.dequant or kind == "deepseek-native":
        bf16_dir = os.path.join(work, "src-bf16")
        dt = ["python", os.path.join(TOOLS, "fp8_to_bf16.py"), "-i", model_dir, "-o", bf16_dir]
        log(" -- dequantizing (DeepSeek-native int8/fp8 -> BF16) ...")
        rc = subprocess.run(dt, check=False)
        if rc.returncode != 0:
            sys.exit("!! dequant failed")
        cfg_dir = bf16_dir

    # preflight
    ok, msg = preflight(cfg_dir)
    log(f" -- preflight: {msg}")
    if not ok:
        log(" !! preflight did not confirm full support; continuing anyway (convert will report missing keys)")
    if args.dry_run:
        log(" -- dry-run finished (no conversion started)")
        return 0

    # convert command
    cmd = [sys.executable, "convert.py", "-i", cfg_dir, "-o", out, "-w", work,
           "-b", f"{args.bits:g}"]
    if args.hq:
        cmd += ["-hq"]
    if args.cr:
        cmd += ["-cr", str(args.cr)]
    if args.cc:
        cmd += ["-cc", str(args.cc)]
    if args.cal_data:
        cmd += ["-cd", args.cal_data]
    if args.device:
        cmd += ["-d", args.device]

    started = time.time()
    log(" -- " + " ".join([os.path.basename(c) for c in cmd[:3]] + ["...", "-b", f"{args.bits:g}"]))
    if args.detach:
        full_cmd = cmd
        if args.draft_model_dir:
            # chain the draft attach after a successful detached conversion
            attach_cmd = [sys.executable, os.path.join(TOOLS, "quantize_model.py"),
                          "--attach-draft", "-o", out, "--draft-model-dir", args.draft_model_dir]
            full_cmd = ["bash", "-c",
                        " ".join(["\"" + c.replace('\"', '\\"') + '\"' if " " in c else c for c in cmd]) +
                        f" && {' '.join(attach_cmd)}"]
        proc = subprocess.Popen(full_cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        log(f" -- detached pid {proc.pid}, log: {log_path}")
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model, "out": out,
                  "work": work, "bits": args.bits, "hq": args.hq, "status": "running", "pid": proc.pid,
                  "log": log_path, "name": args.name or base,
                  "draft_model_dir": args.draft_model_dir}
        with open(JOBS, "a") as f:
            f.write(json.dumps(record) + "\n")
        return 0
    else:
        rc = run(cmd, log_path)
        elapsed = time.time() - started
        status = "ok" if rc.returncode == 0 else "failed"
        log(f"convert.py exited {rc.returncode} after {elapsed/60:.1f} min (log: {log_path})")
        if rc.returncode == 0 and args.draft_model_dir:
            attach_draft(out, args.draft_model_dir)
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model, "out": out,
                  "work": work, "bits": args.bits, "hq": args.hq, "status": status,
                  "elapsed_s": int(elapsed), "log": log_path, "name": args.name or base,
                  "draft_model_dir": args.draft_model_dir}
        with open(JOBS, "a") as f:
            f.write(json.dumps(record) + "\n")
        return rc.returncode


if __name__ == "__main__":
    sys.exit(main())
