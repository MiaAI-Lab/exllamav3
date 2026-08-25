#!/usr/bin/env python3
"""Sanity: wm-1m bundle (YaRN factor 4) + quantized KV cache (-cq 4) with DFlash2 spec-dec.
Verifies: yarn rope active, quant cache layers, generation sane, acceptance recorded."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler

parser = ArgumentParser()
model_init.add_args(parser, add_draft_model_args = True)
args = parser.parse_args([
    "-m", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m",
    "-dm", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm-1m/draft-dflash2",
    "-gs", "110", "-cs", "4096", "-cq", (sys.argv[1] if len(sys.argv) > 1 else "4"),
])
model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
    model_init.init(args, progress = True)

print("\n== rope settings ==")
print(config.rope_settings)
print("target cache layer:", type(cache.layers[list(cache.layers)[0]]).__name__)
print("draft  cache layer:", type(draft_cache.layers[list(draft_cache.layers)[0]]).__name__)
gb = sum(l.storage_size() for l in cache.layers.values()) / 1e9
print(f"target KV storage @ -cs 4096: {gb:.3f} GB -> per 1M tok: {gb * 1e6 / 4096:.1f} GB")

generator = Generator(model, cache, tokenizer, draft_model = draft_model,
                      draft_cache = draft_cache, num_draft_tokens = 7,
                      record_draft_stats = True)
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt)
job = Job(input_ids = input_ids, max_new_tokens = 40, sampler = ArgmaxSampler(), identifier = 0)
generator.enqueue(job)
while generator.num_remaining_jobs():
    generator.iterate()
text = tokenizer.decode(job.sequences[0].sequence_ids.torch_slice(input_ids.shape[-1], None)[0])
print("\n== greedy 40 tok ==")
print(repr(text[:200]))
ds = job.draft_stats
if ds:
    # entries are tuples of accepted counts (DFlash2) or dicts/int (DSpark)
    def _acc(d):
        if isinstance(d, dict):
            return d.get("accepted", 0)
        if isinstance(d, (tuple, list)):
            return sum(x for x in d if isinstance(x, (int, float)))
        return d
    accs = [_acc(d) for d in ds]
    print("accept per round:", accs[:12], "| mean:", sum(accs) / len(accs))
