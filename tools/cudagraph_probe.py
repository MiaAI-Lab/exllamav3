#!/usr/bin/env python3
"""T4.0 probe: CUDA-graph the target verify forward via a model.forward hook.

The generator already feeds forward() from pointer-stable pinned staging buffers
(ids, block_table, cache_seqlens), so a captured graph replays correctly when the
generator refills those buffers each step. This probe validates that assumption:
1. parity: greedy generation, eager vs graphed — token IDs must match exactly
2. speed: decode tok/s with the graph hook off vs on
Recurrent states are snapshotted/restored around warmup+capture (forward advances
them, so warmup reruns are not idempotent).
"""
import sys, os, time, functools
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from argparse import ArgumentParser
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler, ComboSampler

PROBES = {
    "code":  "Write a complete, production-quality LRU cache class in Python "
             "with get, put, and a test suite covering eviction and TTL.",
    "essay": "Write an essay tracing the history of computing from Charles "
             "Babbage's Analytical Engine to modern GPU accelerators.",
}


class GraphForwardHook:
    """Wraps model.forward; captures per (shape, block-width, mode) key after warmup."""

    def __init__(self, model):
        self.model = model
        self.orig = model.forward
        self.enabled = False
        self.graphs = {}         # key -> CUDAGraph
        self.outputs = {}        # key -> captured output tensor
        self.seen = {}           # key -> call count
        self.capture_args = None
        self.stats = {"replays": 0, "eager": 0, "captures": 0}
        model.forward = self._call

    def _key(self, input_ids, params):
        return (tuple(input_ids.shape),
                params.get("block_table").shape[-1] if params.get("block_table") is not None else 0,
                params.get("recurrent_history") is not None)

    @staticmethod
    def _snapshot_states(params):
        states = params.get("recurrent_states")
        if not states: return None
        saved = []
        for st in states:
            snap = {}
            for k, v in vars(st).items():
                snap[k] = v.detach().clone() if torch.is_tensor(v) else v
            saved.append((st, snap))
        return saved

    @staticmethod
    def _restore_states(saved):
        if not saved: return
        for st, snap in saved:
            for k, v in snap.items():
                cur = getattr(st, k)
                if torch.is_tensor(v): cur.copy_(v)
                else: setattr(st, k, v)

    def _run_forward(self, input_ids, params):
        with torch.inference_mode():
            return self.orig(input_ids = input_ids, params = params)

    def _call(self, input_ids, params = None):
        if (not self.enabled or params is None or
                params.get("attn_mode") != "flash_attn" or
                input_ids.device.type != "cpu" or
                params.get("block_table") is None or
                params.get("cache_seqlens") is None or
                params.get("cache_seqlens").device.type != "cpu"):
            self.stats["eager"] += 1
            return self.orig(input_ids = input_ids, params = params)

        key = self._key(input_ids, params)
        n = self.seen.get(key, 0) + 1
        self.seen[key] = n

        if key in self.graphs:
            self.graphs[key].replay()
            self.stats["replays"] += 1
            return self.outputs[key]

        if n < 3:
            # steady-state warmup observations (eager, real side effects)
            self.stats["eager"] += 1
            return self.orig(input_ids = input_ids, params = params)

        # capture: rerun forward a few times (idempotent for KV writes at fixed
        # seqlens, but NOT for recurrent states -> snapshot/restore)
        torch.cuda.synchronize()
        saved = self._snapshot_states(params)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._run_forward(input_ids, params)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._run_forward(input_ids, params)
            torch.cuda.synchronize()
            self.graphs[key] = g
            self.outputs[key] = out
            self.stats["captures"] += 1
        finally:
            self._restore_states(saved)
            torch.cuda.synchronize()
        g.replay()
        self.stats["replays"] += 1
        return out

    def purge(self):
        self.graphs.clear()
        self.outputs.clear()
        self.seen.clear()


def run(generator, tokenizer, prompt, max_new, greedy = True, seed = 1234):
    torch.manual_seed(0)
    input_ids = tokenizer.encode(prompt)
    if greedy:
        sampler = ArgmaxSampler()
    else:
        sampler = ComboSampler(temperature = 0.6, top_k = 20, top_p = 0.95)
    job = Job(input_ids = input_ids, max_new_tokens = max_new,
              stop_conditions = ["<|im_end|>", tokenizer.eos_token_id],
              sampler = sampler, seed = seed)
    generator.enqueue(job)
    seq_len0 = input_ids.shape[-1]
    t_first = None; len_first = seq_len0; t_end = None; len_end = seq_len0
    while generator.num_remaining_jobs():
        generator.iterate()
        L = len(job.sequences[0].sequence_ids)
        if L > seq_len0 and t_first is None:
            t_first = time.perf_counter(); len_first = L
        t_end = time.perf_counter(); len_end = L
    ids = job.sequences[0].sequence_ids.torch_slice(seq_len0, None)[0]
    toks = len_end - len_first
    dt = t_end - t_first
    return ids, (toks / dt if dt > 0 else 0.0)


def main():
    ap = ArgumentParser()
    ap.add_argument("--mode", default = "both", choices = ["parity", "speed", "both"])
    ap.add_argument("--max-new", type = int, default = 160)
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
    hook = GraphForwardHook(model)

    if args.mode in ("parity", "both"):
        ids_eager, _ = run(generator, tokenizer, PROBES["code"], args.max_new, greedy = True)
        hook.purge()
        hook.enabled = True
        ids_graph, _ = run(generator, tokenizer, PROBES["code"], args.max_new, greedy = True)
        hook.enabled = False
        same = torch.equal(ids_eager, ids_graph)
        n = min(len(ids_eager), len(ids_graph))
        first_diff = next((i for i in range(n) if ids_eager[i] != ids_graph[i]), None)
        print(f"\n== PARITY (greedy {args.max_new} tok): "
              f"{'EXACT MATCH' if same else f'DIFFER (len {len(ids_eager)} vs {len(ids_graph)}, first at {first_diff})'}")
        print(f"   hook stats: {hook.stats}")
        assert same, "graph parity FAILED"

    if args.mode in ("speed", "both"):
        print("\n== SPEED (T0.6, decode-only tok/s) ==")
        for name, prompt in PROBES.items():
            for label, enabled in (("eager ", False), ("graph ", True)):
                hook.purge()
                hook.enabled = enabled
                tps = []
                for _ in range(2):
                    _, t = run(generator, tokenizer, prompt, 384, greedy = False)
                    tps.append(t)
                print(f"  {name:5s} {label}: " + "  ".join(f"{x:6.2f}" for x in tps) + " tok/s")
        hook.enabled = False
        print(f"   hook stats: {hook.stats}")


if __name__ == "__main__":
    main()
