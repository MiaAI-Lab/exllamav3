#!/usr/bin/env python3
"""Measure per-step decode breakdown: wall vs target-forward GPU time vs draft GPU time,
via CUDA events hooked around model.forward + draft forward. Reports GPU-busy fraction and
effective aggregate bandwidth (weights+draft read once per step as the byte model)."""
import sys, os, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch
from argparse import ArgumentParser
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler.presets import ComboSampler

PROMPT = ("Write a complete, production-quality LRU cache class in Python "
          "with get, put, and a test suite covering eviction and TTL.")

class EventTimer:
    def __init__(self, model, draft_model):
        self.t_target = 0.0; self.t_draft = 0.0; self.n_t = 0; self.n_d = 0
        self._install(model, "target"); self._install(draft_model, "draft")
    def _install(self, m, which):
        orig = m.forward; timer = self
        def fwd(input_ids=None, params=None):
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = orig(input_ids=input_ids, params=params)
            ev1.record(); ev1.synchronize()
            if which == "target":
                timer.t_target += ev0.elapsed_time(ev1) / 1e3; timer.n_t += 1
            else:
                timer.t_draft += ev0.elapsed_time(ev1) / 1e3; timer.n_d += 1
            return out
        m.forward = fwd

def main():
    ap = ArgumentParser()
    model_init.add_args(ArgumentParser(), add_draft_model_args=True)  # noqa
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
    timer = EventTimer(model, draft_model)

    sampler = ComboSampler(temperature=0.6, top_k=20, top_p=0.95)
    ids = tokenizer.encode(PROMPT)
    job = Job(input_ids=ids, max_new_tokens=240, stop_conditions=["<|im_end|>"],
              sampler=sampler, seed=1234)
    generator.enqueue(job)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    first = None
    while generator.num_remaining_jobs():
        generator.iterate()
        if first is None and len(job.sequences[0].sequence_ids) > ids.shape[-1]:
            torch.cuda.synchronize(); first = time.perf_counter()
    torch.cuda.synchronize(); t1 = time.perf_counter()

    wall = t1 - first
    steps = timer.n_t
    # steady state only: subtract the first forward (prefill-ish warmup inside events) approximately
    print(f"\nwall decode: {wall:.2f} s, target fwds: {timer.n_t}, draft fwds: {timer.n_d}")
    print(f"target GPU: {timer.t_target:.2f} s ({timer.t_target/steps*1e3:.1f} ms/step)")
    print(f"draft  GPU: {timer.t_draft:.2f} s ({timer.t_draft/steps*1e3:.1f} ms/step)")
    gpu = timer.t_target + timer.t_draft
    print(f"GPU-busy fraction: {gpu/wall*100:.1f}%   (gap = {wall-gpu:.2f} s)")
    print(f"decode tok/s (wall): {(len(job.sequences[0].sequence_ids)-ids.shape[-1]-1)/wall:.2f}")

if __name__ == "__main__":
    main()
