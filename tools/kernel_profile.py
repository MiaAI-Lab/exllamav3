#!/usr/bin/env python3
"""Profile ONE steady-state verify forward with torch.profiler (CUDA kernels) and print
the per-kernel GPU-time table."""
import sys, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch
from argparse import ArgumentParser
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler.presets import ComboSampler

PROMPT = ("Write a complete, production-quality LRU cache class in Python "
          "with get, put, and a test suite covering eviction and TTL.")

STEP = int(os.environ.get("PROFILE_AT", "30"))
WHAT = os.environ.get("PROFILE_WHAT", "target")   # target | draft | verify_accept

def main():
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args=True)
    margs = parser.parse_args([
        "-m", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m",
        "-dm", os.environ.get("DM", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m/draft-dflash2"),
        "-gs", "110", "-cs", "1048576", "-cq", "nvfp4",
    ])
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(margs, progress=True)
    generator = Generator(model, cache, tokenizer, draft_model=draft_model,
                          draft_cache=draft_cache, num_draft_tokens=7)

    prof_model = draft_model if WHAT == "draft" else model
    orig = prof_model.forward
    state = {"n": 0, "prof": None}
    def fwd(input_ids=None, params=None):
        state["n"] += 1
        if state["n"] == STEP:
            p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU])
            state["prof"] = p
            p.__enter__()
            out = orig(input_ids=input_ids, params=params)
            p.__exit__(None, None, None)
            return out
        return orig(input_ids=input_ids, params=params)
    prof_model.forward = fwd

    sampler = ComboSampler(temperature=0.6, top_k=20, top_p=0.95)
    ids = tokenizer.encode(PROMPT)
    job = Job(input_ids=ids, max_new_tokens=120, stop_conditions=["<|im_end|>"],
              sampler=sampler, seed=1234)
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        generator.iterate()
        if state["prof"] is not None:
            break

    p = state["prof"]
    print("\n==== CUDA kernel table, one %s forward (step %d) ====" % (WHAT, STEP))
    evs = p.key_averages()
    rows = [(e.self_device_time_total, e.count, e.key) for e in evs if e.self_device_time_total > 0]
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    for us, n, k in rows[:35]:
        print(f"{us/1e3:9.2f} ms  n={n:<4d} {k[:110]}")
    print(f"TOTAL device: {tot/1e3:.2f} ms")

if __name__ == "__main__":
    main()
