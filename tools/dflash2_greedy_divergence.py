#!/usr/bin/env python3
"""Greedy divergence probe: dflash_run (batched 8-row verify) vs plain
incremental HF generate, both greedy, same target.

For each divergence index, reports the incremental path's top-2 logit gap at
that step. Tiny gaps (< ~0.1) => argmax flips under bf16 kernel/tiling drift
between batch shapes; large gaps at early positions => real bug.
"""
import sys, os, json, torch
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo)

from transformers import AutoModelForCausalLM, AutoTokenizer
from dflash.model import DFlash2DraftModel
from tools.dflash2_reference_baselines import (PROMPTS, dflash_run,
    _make_cache, extract_context_feature)
import dflash.model as DM

DEV = "cuda:0"
TDIR = "test_models/sources/Qwen3.8-27B"
DDIR = "test_models/sources/Qwen3.8-27B-DFlash2"

tok = AutoTokenizer.from_pretrained(TDIR)
target = AutoModelForCausalLM.from_pretrained(
    TDIR, torch_dtype=torch.bfloat16, device_map=DEV)
target.eval()
draft = DFlash2DraftModel.from_pretrained(DDIR, torch_dtype=torch.bfloat16,
                                          device_map=DEV).eval()
print("loaded")

@torch.inference_mode()
def plain_greedy_ids_and_gaps(input_ids, n, min_pos):
    """Incremental greedy; returns ids [n] and top-2 gaps for pos >= min_pos."""
    past = _make_cache(target.config)
    out = target(input_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
    ids, gaps = [], [None] * min_pos
    logits = out.logits[0, -1].float()
    for i in range(n):
        t2 = torch.topk(logits, 2)
        nxt = int(t2.indices[0])
        ids.append(nxt)
        gaps.append((float(t2.values[0] - t2.values[1])) if i >= min_pos else None)
        out = target(torch.tensor([[nxt]], device=DEV), past_key_values=past,
                     use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1].float()
    return ids, gaps

@torch.inference_mode()
def main():
    res = {}
    for pi in range(3):
        ids = tok(PROMPTS[pi], return_tensors="pt").input_ids.to(DEV)
        spec_out, acc = dflash_run(draft, target, ids, 128, None, 0.0)
        spec = spec_out[0, ids.shape[1]:].tolist()
        plain, gaps = plain_greedy_ids_and_gaps(ids, 128, ids.shape[1])
        n = min(len(spec), len(plain))
        mism = [i for i in range(n) if spec[i] != plain[i]]
        first = mism[0] if mism else None
        # cumulative drift distance: how many tokens match
        match_run = 0
        while match_run < n and spec[match_run] == plain[match_run]:
            match_run += 1
        gap_at = gaps[first] if first is not None and first < len(gaps) and gaps[first] is not None else None
        print(f"p{pi}: match_run={match_run}, mismatches={len(mism)}/{n}, "
              f"first_div_idx={first}, top2_gap_at_first={gap_at}")
        res[f"p{pi}"] = dict(match_run=match_run, n_mismatch=len(mism),
                             first=first, gap=gap_at,
                             next_gaps=[g for g in gaps[first:first+5]] if first is not None else None)
    json.dump(res, open(os.path.join(repo,
        "test_models/dspark-evals/dflash2/greedy_divergence.json"), "w"), indent=1)

main()
