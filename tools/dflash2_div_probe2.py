#!/usr/bin/env python3
"""Focused greedy divergence probe (one prompt): at the first token where the
dflash loop (batched 8-row verify) and incremental greedy disagree, print the
incremental path's top-2 logit gap and both top-5 logit rows, to distinguish
near-tie bf16 batch-shape flips from structural bugs."""
import sys, os, torch
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo)

from transformers import AutoModelForCausalLM, AutoTokenizer
from dflash.model import DFlash2DraftModel
from tools.dflash2_reference_baselines import PROMPTS, dflash_run, _make_cache

DEV = "cuda:0"
TDIR = "test_models/sources/Qwen3.8-27B"
DDIR = "test_models/sources/Qwen3.8-27B-DFlash2"

tok = AutoTokenizer.from_pretrained(TDIR)
target = AutoModelForCausalLM.from_pretrained(TDIR, torch_dtype=torch.bfloat16,
                                              device_map=DEV).eval()
draft = DFlash2DraftModel.from_pretrained(DDIR, torch_dtype=torch.bfloat16,
                                          device_map=DEV).eval()

@torch.inference_mode()
def plain_greedy_logits(input_ids, n):
    """Incremental greedy; returns (ids, per-step top5 logit rows)."""
    past = _make_cache(target.config)
    out = target(input_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
    logits = out.logits[0, -1].float()
    ids, tops = [], []
    for _ in range(n):
        t5 = torch.topk(logits, 5)
        nxt = int(t5.indices[0])
        ids.append(nxt)
        tops.append(t5.values.cpu())
        out = target(torch.tensor([[nxt]], device=DEV), past_key_values=past,
                     use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1].float()
    return ids, tops

@torch.inference_mode()
def main():
    pi = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ids = tok(PROMPTS[pi], return_tensors="pt").input_ids.to(DEV)
    spec_out, acc = dflash_run(draft, target, ids, 128, None, 0.0)
    spec = spec_out[0, ids.shape[1]:].tolist()
    plain, tops = plain_greedy_logits(ids, 128)
    n = min(len(spec), len(plain))
    first = next((i for i in range(n) if spec[i] != plain[i]), None)
    print(f"p{pi}: first divergence at token {first}")
    if first is not None:
        t = tops[first]
        gap = float(t[0] - t[1])
        print(f"incremental top-5 logits at that step: {[round(v, 3) for v in t.tolist()]}")
        print(f"top-2 gap = {gap:.4f}  (near-tie if < ~0.1)")
        print(f"spec token {spec[first]} vs plain token {plain[first]}")

main()
