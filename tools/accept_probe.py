#!/usr/bin/env python3
"""Acceptance-chain stats: greedy vs sampled decode on same probe. Reports accept length
distribution, tok/s, and per-step wall — quantifies what rejection-sampling acceptance
could recover for sampled decode."""
import sys, os, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch
from argparse import ArgumentParser
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler, ComboSampler

PROMPT = ("Write a complete, production-quality LRU cache class in Python "
          "with get, put, and a test suite covering eviction and TTL.")

def main():
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args=True)
    margs = parser.parse_args([
        "-m", os.environ.get("TM", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m"),
        "-dm", os.environ.get("DM", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m/draft-dflash2"),
        "-gs", "110", "-cs", "1048576", "-cq", "nvfp4",
    ])
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(margs, progress=True)
    generator = Generator(model, cache, tokenizer, draft_model=draft_model,
                          draft_cache=draft_cache, num_draft_tokens=7, record_draft_stats=True)

    from collections import Counter
    import hashlib
    variants = [("greedy  ", ArgmaxSampler())]
    t = float(os.environ.get("DRAFT_T", "0.6"))
    if t > 0:
        variants.append((f"T{t:<5}", ComboSampler(temperature=t, top_k=20, top_p=0.95)))
    for label, sampler in variants:
        for rep in range(2):
            torch.manual_seed(1234)
            ids = tokenizer.encode(PROMPT)
            job = Job(input_ids=ids, max_new_tokens=240, stop_conditions=["<|im_end|>"],
                      sampler=sampler, seed=1234)
            generator.enqueue(job)
            steps = Counter()
            first = None
            while generator.num_remaining_jobs():
                generator.iterate()
                if first is None and len(job.sequences[0].sequence_ids) > ids.shape[-1]:
                    torch.cuda.synchronize(); first = time.perf_counter()
            torch.cuda.synchronize()
            wall = time.perf_counter() - first
            # acceptance stats from job draft_stats
            ds = getattr(job, "draft_stats", None) or []
            acc = [s[2] if isinstance(s, (tuple, list)) else s for s in ds]
            acc = [int(a) for a in acc]
            n = len(acc); mean = sum(acc)/n if n else 0
            hist = Counter(acc)
            toks = len(job.sequences[0].sequence_ids) - ids.shape[-1] - 1
            if getattr(generator, "rs_debug", None):
                dbg = generator.rs_debug
                import collections as _c
                by_pos = _c.defaultdict(list)
                for (pi, pd_, qd_, ac) in dbg: by_pos[pi].append((pd_, qd_, ac))
                print("    RS debug per draft position: n, accept_rate, mean p_d, mean q_d, frac(p_d>0)")
                for pi in sorted(by_pos):
                    v = by_pos[pi]
                    ar = sum(1 for x in v if x[2]) / len(v)
                    mp = sum(x[0] for x in v) / len(v)
                    mq = sum(x[1] for x in v) / len(v)
                    fp = sum(1 for x in v if x[0] > 0) / len(v)
                    print(f"      pos {pi}: n={len(v):4d} acc={ar:.3f} mean_p={mp:.4f} mean_q={mq:.4f} p>0: {fp:.3f}")
                generator.rs_debug.clear()
            import hashlib as _h
            h = _h.sha1(
                job.sequences[0].sequence_ids.torch_slice(ids.shape[-1], None)[0]
                .cpu().numpy().tobytes()).hexdigest()[:10]
            print(f"{label} r{rep}: steps={n} accept_mean={mean:.3f} "
                  f"hist={ {k: hist[k] for k in sorted(hist)} } "
                  f"toks={toks} tok/s={toks/wall:.2f} sha={h}")

if __name__ == "__main__":
    main()
