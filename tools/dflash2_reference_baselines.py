#!/usr/bin/env python3
"""M0.4 — DFlash2 reference generation baselines (dspark-imp.md §M0.4).

GPU required (server must be down). Runs the reference dflash cycle loop
(transcribed, same as tools/dflash2_cache_trace.py) with the real HF bf16
Qwen3.8-27B target + the DFlash2 draft checkpoint:

  - 10 fixed prompts x {greedy, T=0.6, T=1.0} x 128 new tokens
  - per-run acceptance lengths -> mean vs published 4.80 (their convention:
    T=1.0, top-p 0.95, top-k 20)
  - greedy lossless sanity: dflash greedy token stream == plain target
    greedy for every prompt
  - M1 parity fixtures (2 prompts, greedy): per-cycle target_hidden,
    draft_hidden, proposal (tokens/indices/probs), accepted length

Run:  .venv/bin/python tools/dflash2_reference_baselines.py
Out:  test_models/dspark-evals/dflash2/baselines_<ts>.json
      test_models/dspark-evals/dflash2/parity_fixtures/  (*.pt)
"""

import sys, os, json, time, argparse
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import dflash.model as DM
from dflash.model import (
    DFlash2DraftModel, extract_context_feature, _make_cache, _crop_to,
    _sampling_probs, _sample_probs, _rejection_sample, sample,
    _raw_input_embeddings, _output_head,
)

TARGET_DIR = os.path.join(repo_root, "test_models/sources/Qwen3.8-27B")
DRAFT_DIR = os.path.join(repo_root, "test_models/sources/Qwen3.8-27B-DFlash2")
OUT_DIR = os.path.join(repo_root, "test_models/dspark-evals/dflash2")
FIX_DIR = os.path.join(OUT_DIR, "parity_fixtures")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed?\nA:",
    "The theory of general relativity describes how gravity",
    "import numpy as np\n\ndef softmax(x):\n    ",
    "In a surprising turn of events, the committee decided to",
    "The three laws of robotics, as formulated by Isaac Asimov, are",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "Water boils at a lower temperature at high altitude because",
    "Once upon a time in a small coastal village, there lived",
]

@torch.inference_mode()
def dflash_run(model, target, input_ids, max_new_tokens, stop_ids, temperature,
               top_p=1.0, top_k=0, fixture_cb=None):
    """Reference cycle loop (identical to dflash2_cache_trace.traced_generate,
    minus mask instrumentation, plus optional per-cycle fixture callback)."""
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size
    output_ids = torch.full((1, max_length + 1), model.mask_token_id,
                            dtype=torch.long, device=target.device)
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_target = _make_cache(target.config)
    past_draft = _make_cache(model.config)

    output = target(input_ids,
                    position_ids=position_ids[:, :num_input_tokens],
                    past_key_values=past_target, use_cache=True,
                    logits_to_keep=1, output_hidden_states=block_size > 1)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(
        output.logits, temperature, top_p, top_k)
    target_hidden = extract_context_feature(
        output.hidden_states, model.target_layer_ids)
    _crop_to(past_target, num_input_tokens)

    acceptance = []
    start = num_input_tokens
    stop_tokens = torch.tensor(stop_ids, device=target.device) if stop_ids else None
    stopped = (stop_tokens is not None
               and torch.isin(output_ids[:, start], stop_tokens).any())
    while start + 1 < max_length and not stopped:
        verify_size = min(block_size, max_length - start)
        block_output_ids = output_ids[:, start:start + verify_size].clone()
        block_position_ids = position_ids[:, start:start + verify_size]
        ctx_len = target_hidden.shape[1]
        if verify_size > 1:
            noise_embedding = _raw_input_embeddings(
                target, block_output_ids,
                float(DM._draft_value(model.config, "input_embedding_scale", 1.0)))
            draft_hidden = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, start - ctx_len:start + verify_size],
                past_key_values=past_draft,
                use_cache=True)[:, 1 - verify_size:, :]
            _crop_to(past_draft, start)
            draft_tokens, draft_indices, draft_probs = model.propose(
                draft_hidden, block_output_ids[:, 0], _output_head(target),
                temperature)
            block_output_ids[:, 1:] = draft_tokens
        output = target(block_output_ids, position_ids=block_position_ids,
                        past_key_values=past_target, use_cache=True,
                        output_hidden_states=verify_size > 1)
        if temperature > 0:
            target_probs = _sampling_probs(output.logits, temperature, top_p, top_k)
            if verify_size > 1:
                acceptance_length, bonus = _rejection_sample(
                    block_output_ids[:, 1:], target_probs, draft_probs,
                    draft_indices)
            else:
                acceptance_length = 0
                bonus = _sample_probs(target_probs[:, -1])[0]
        else:
            posterior = torch.argmax(output.logits, dim=-1)
            acceptance_length = (block_output_ids[:, 1:] ==
                                 posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            bonus = posterior[:, acceptance_length][0]
        output_ids[:, start:start + acceptance_length + 1] = \
            block_output_ids[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = bonus
        produced = min(acceptance_length + 1, max_length - start - 1)
        if stop_tokens is not None:
            stop_idx = torch.isin(output_ids[0, start + 1:start + produced + 1],
                                  stop_tokens).nonzero(as_tuple=True)[0]
            if stop_idx.numel() > 0:
                produced = stop_idx[0].item() + 1
                stopped = True
        start += produced
        _crop_to(past_target, start)
        acceptance.append(produced)
        prev_hidden = target_hidden                     # ctx that fed this cycle
        if verify_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, model.target_layer_ids)[:, :produced, :]
        if fixture_cb is not None:
            fixture_cb(cycle=len(acceptance) - 1, ctx_in=prev_hidden,
                       draft_hidden=draft_hidden, tokens=draft_tokens,
                       indices=draft_indices, probs=draft_probs,
                       accepted=acceptance_length, produced=produced,
                       block_ids=block_output_ids.clone())
    return output_ids[:, :min(start + 1, max_length)], acceptance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--fixture-prompts", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(FIX_DIR, exist_ok=True)
    dev = "cuda"
    print("== loading target (HF bf16) ...", flush=True)
    t0 = time.time()
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_DIR, torch_dtype=torch.bfloat16, device_map=dev)
    target.eval()
    tok = AutoTokenizer.from_pretrained(TARGET_DIR)
    print(f"   target ready in {time.time()-t0:.0f}s", flush=True)
    print("== loading draft ...", flush=True)
    draft = DFlash2DraftModel.from_pretrained(DRAFT_DIR, torch_dtype=torch.bfloat16,
                                              device_map=dev)
    draft.eval()
    print(f"   draft ready; block_size={draft.block_size}", flush=True)

    stop_ids = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.eos_token_id:
        stop_ids.append(im_end)

    results = {"prompts": PROMPTS, "max_new_tokens": args.max_new_tokens, "runs": {}}
    modes = [("greedy", 0.0, 1.0, 0), ("t06", 0.6, 0.95, 20), ("t10", 1.0, 0.95, 20)]

    for mode, T, tp, tk in modes:
        per_prompt = []
        for pi, prompt in enumerate(PROMPTS):
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            torch.manual_seed(42 + pi)                 # deterministic sampling
            t0 = time.time()
            out, acc = dflash_run(draft, target, ids, args.max_new_tokens,
                                  stop_ids, T, tp, tk,
                                  fixture_cb=(lambda **kw: torch.save(
                                      {k: (v.cpu() if torch.is_tensor(v) else v)
                                       for k, v in kw.items()},
                                      os.path.join(FIX_DIR, f"p{pi}_{mode}_c{kw['cycle']:03d}.pt")))
                                  if (mode == "greedy" and pi < args.fixture_prompts)
                                  else None)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            per_prompt.append({
                "prompt": prompt, "new_tokens": int(out.shape[1] - ids.shape[1]),
                "ids": out[0, ids.shape[1]:].tolist(),
                "mean_acceptance": sum(acc) / len(acc) if acc else 0.0,
                "acceptance": acc, "secs": round(time.time() - t0, 1),
                "text_head": text[:120],
            })
            print(f"  [{mode}] p{pi}: {out.shape[1]-ids.shape[1]} tok, "
                  f"acc={per_prompt[-1]['mean_acceptance']:.2f}, "
                  f"{per_prompt[-1]['secs']}s", flush=True)
        results["runs"][mode] = per_prompt

    # ---- greedy lossless sanity: plain target greedy == dflash greedy ----
    print("== greedy lossless sanity (plain target greedy) ...", flush=True)
    all_match = True
    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        plain = target.generate(ids, max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                eos_token_id=stop_ids, pad_token_id=stop_ids[0])
        spec = results["runs"]["greedy"][pi]["ids"]
        plain_out = plain[0, ids.shape[1]:].tolist()
        n = min(len(plain_out), len(spec))
        match = plain_out[:n] == spec[:n]
        all_match &= match
        print(f"   p{pi}: {'MATCH' if match else 'MISMATCH'} ({n} tok)", flush=True)
        results["runs"]["greedy"][pi]["greedy_match"] = bool(match)
    print(f"== greedy lossless: {'ALL MATCH' if all_match else 'MISMATCH FOUND'}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"baselines_{ts}.json")
    # strip bulky fields from json (acceptance kept)
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"== wrote {path}")

    for mode, T, tp, tk in modes:
        accs = [p["mean_acceptance"] for p in results["runs"][mode]]
        print(f"   {mode}: mean acceptance {sum(accs)/len(accs):.3f}")

if __name__ == "__main__":
    main()
