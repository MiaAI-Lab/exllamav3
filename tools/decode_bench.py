#!/usr/bin/env python3
"""Decode tok/s bench for the serving lane (wm-1m + DFlash2 + -cq nvfp4).

Steady-state decode tok/s (excludes prefill) on fixed code/essay probes at
serving sampling (T0.6/k20/p0.95, thinking on). Interleaved A/B of CPU affinity
(default vs EXL3_AFFINITY, default "5-9,15-19") to cancel power-cap drift.
Same seed per run: A/B outputs must match bit-exactly (GPU math identical).
"""
import sys, os, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# affinity FIRST, before torch threads spawn
def parse_cpuspec(spec):
    cpus = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); cpus.update(range(int(a), int(b) + 1))
        elif part.strip():
            cpus.add(int(part))
    return cpus

ORIG_AFFINITY = os.sched_getaffinity(0)
PIN_SPEC = os.environ.get("EXL3_AFFINITY", "5-9,15-19")
PIN = parse_cpuspec(PIN_SPEC) if PIN_SPEC else None

import torch
from argparse import ArgumentParser
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler.presets import ComboSampler

PROBES = {
    "code":  "Write a complete, production-quality LRU cache class in Python "
             "with get, put, and a test suite covering eviction and TTL.",
    "essay": "Write an essay tracing the history of computing from Charles "
             "Babbage's Analytical Engine to modern GPU accelerators.",
}

def run_probe(generator, tokenizer, name, prompt, max_new = 512):
    torch.manual_seed(0)
    input_ids = tokenizer.encode(prompt)
    sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)
    job = Job(input_ids = input_ids, max_new_tokens = max_new,
              stop_conditions = ["<|im_end|>", tokenizer.eos_token_id],
              sampler = sampler, seed = 1234)
    generator.enqueue(job)
    seq_len0 = input_ids.shape[-1]
    t_first = None; len_first = seq_len0; t_end = None; len_end = seq_len0
    while generator.num_remaining_jobs():
        generator.iterate()
        L = len(job.sequences[0].sequence_ids)   # includes prompt; SeqTensor __len__
        if L > seq_len0 and t_first is None:
            t_first = time.perf_counter(); len_first = L
        t_end = time.perf_counter(); len_end = L
    toks = len_end - len_first
    dt = t_end - t_first
    acc = None
    try:
        ds = job.draft_stats
        def _a(d):
            if isinstance(d, dict): return d.get("accepted", 0)
            if isinstance(d, (tuple, list)): return sum(x for x in d if isinstance(x, (int, float)))
            return d
        if ds: acc = sum(_a(d) for d in ds) / len(ds)
    except Exception:
        pass
    text = tokenizer.decode(job.sequences[0].sequence_ids.torch_slice(seq_len0, None)[0])
    return toks / dt if dt > 0 else 0.0, toks, dt, acc, text

def main():
    ap = ArgumentParser()
    ap.add_argument("--runs", type = int, default = 3)
    ap.add_argument("--profile", action = "store_true")
    args = ap.parse_args()

    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = True)
    margs = parser.parse_args([
        "-m", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m",
        "-dm", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m/draft-dflash2",
        "-gs", "110", "-cs", "1048576", "-cq", "nvfp4",
    ])
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(margs, progress = True)
    generator = Generator(model, cache, tokenizer, draft_model = draft_model,
                          draft_cache = draft_cache, num_draft_tokens = 7)

    variants = [("unpinned", ORIG_AFFINITY), ("pinned", PIN)]
    variants = [(n, a) for n, a in variants if a]
    results = {n: {p: [] for p in PROBES} for n, _ in variants}
    texts = {n: {} for n, _ in variants}

    for r in range(args.runs):
        for name, prompt in PROBES.items():
            for vname, aff in variants:
                os.sched_setaffinity(0, aff)
                tps, toks, dt, acc, text = run_probe(generator, tokenizer, name, prompt)
                results[vname][name].append(tps)
                texts[vname].setdefault(name, []).append(text[:60])
                print(f"run {r} {vname:8s} {name:5s}: {tps:6.2f} tok/s  "
                      f"({toks} tok in {dt:.2f}s, accept {acc:.2f})" if acc is not None else
                      f"run {r} {vname:8s} {name:5s}: {tps:6.2f} tok/s", flush = True)

    print("\n== summary (median) ==")
    import statistics as st
    for vname, _ in variants:
        for name in PROBES:
            xs = results[vname][name]
            print(f"{vname:8s} {name:5s}: median {st.median(xs):6.2f}  "
                  f"[{', '.join(f'{x:.2f}' for x in xs)}]")
    if args.profile:
        import cProfile, pstats
        os.sched_setaffinity(0, PIN or ORIG_AFFINITY)
        pr = cProfile.Profile(); pr.enable()
        run_probe(generator, tokenizer, "code", PROBES["code"])
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(25)

if __name__ == "__main__":
    main()
