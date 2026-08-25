#!/usr/bin/env python3
"""Long-context passkey gate: needle recall at depth, with YaRN + quantized KV.
Usage: python tools/longctx_passkey.py <ctx_tokens> <depths_pct_csv> [-cq bits]
Filler: wikitext-2 test stream (distinct natural text)."""
import sys, os, random, string, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler

def main():
    ap = ArgumentParser()
    model_init.add_args(ap, add_draft_model_args = True)
    ap.add_argument("--ctx", type = int, default = 300000)
    ap.add_argument("--depths", type = str, default = "10,50,90")
    args = ap.parse_args([
        "-m", os.environ.get("TM", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m"),
        "-gs", "110", "-cs", "1048576", "-cq", (sys.argv[3] if len(sys.argv) > 3 else "8"),
        "--ctx", sys.argv[1] if len(sys.argv) > 1 else "300000",
        "--depths", sys.argv[2] if len(sys.argv) > 2 else "10,50,90",
    ])
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "test")

    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(args, progress = True)
    print(f"rope: {config.rope_settings.rope_scaling}")
    generator = Generator(model, cache, tokenizer)

    # Build filler token stream from wikitext
    filler_ids = []
    for row in ds:
        t = row["text"].strip()
        if t:
            filler_ids += tokenizer.encode(t)[0].tolist()
        if len(filler_ids) > args.ctx + 4096:
            break
    print(f"filler tokens: {len(filler_ids)}", flush = True)

    random.seed(0)
    depths = [int(d) for d in args.depths.split(",")]
    results = []
    for d in depths:
        key = "".join(random.choices(string.ascii_uppercase, k = 6))
        needle = tokenizer.encode(f"The passkey for the vault is {key}. Remember it.\n")[0].tolist()
        question = tokenizer.encode("\n\nWhat is the passkey for the vault?")[0].tolist()
        pos = min(int((args.ctx - 1024) * d / 100), args.ctx - 1024)
        ids = filler_ids[:pos] + needle + filler_ids[pos : args.ctx - len(needle) - len(question)] + question
        input_ids = torch.tensor(ids, dtype = torch.long).unsqueeze(0)
        t0 = time.time(); print(f"prefill ~{len(ids)} tok ...", flush = True)
        job = Job(input_ids = input_ids, max_new_tokens = 16, sampler = ArgmaxSampler(), identifier = d)
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            generator.iterate()
        out = tokenizer.decode(job.sequences[0].sequence_ids.torch_slice(len(ids), None)[0])
        ok = key in out.upper()
        results.append((d, key, ok))
        print(f"depth {d:3d}%  ctx {len(ids)} tok  key {key}  -> {'PASS' if ok else 'FAIL'}  "
              f"({time.time() - t0:.0f}s)  out={out[:40]!r}")

    npass = sum(1 for _, _, ok in results if ok)
    print(f"\nPASSKEY GATE: {npass}/{len(results)} at ctx target {args.ctx}")

import torch
if __name__ == "__main__":
    main()
