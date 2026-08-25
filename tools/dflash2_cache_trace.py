#!/usr/bin/env python3
"""M0.3 — DFlash2 reference cache/position/mask trace (dspark-imp.md §M0.3).

Server-safe (CPU only). Runs the reference draft cycle loop — transcribed
line-for-line from ``dflash.model.dflash_generate`` — against a stub target
that provides the exact interfaces the reference touches (embeddings,
output head, hidden states at 5 layers, its own DynamicCache), and logs the
draft-KV-cache/position/mask state per cycle to JSONL.

The point (plan risk #1): pin down the draft cache positional semantics —
what stays cached across cycles, which positions the ctx/noise parts cover,
how the bilateral window mask is shaped — so the exl3 port can reproduce
these logs exactly.

Modes:
  short  — prompt 32, greedy (T=0)
  shortT — prompt 32, sampled (T=1.0; exercises _rejection_sample q-rows)
  long   — prompt 2100, greedy (exercises bilateral sliding-window masking)

Run: .venv/bin/python tools/dflash2_cache_trace.py [short|shortT|long|all]
Out: test_models/dspark-evals/dflash2/cache_trace_<mode>.jsonl (+ summary)
"""

import sys, os, json, argparse
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch
from types import SimpleNamespace
from transformers import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import dflash.model as DM
from dflash.model import (
    DFlash2DraftModel, extract_context_feature, _make_cache, _crop_to,
    _sampling_probs, _sample_probs, _rejection_sample, sample,
    _raw_input_embeddings, _output_head, _attention_mask,
)

CKPT = os.path.join(repo_root, "test_models/sources/Qwen3.8-27B-DFlash2")
OUT_DIR = os.path.join(repo_root, "test_models/dspark-evals/dflash2")

# ---------------------------------------------------------------------------
# Stub target: exact interfaces, garbage semantics, honest cache lengths.
# ---------------------------------------------------------------------------

class StubTarget(torch.nn.Module):
    """Interface-compatible stand-in for the 64-layer HF Qwen3.8 target.

    Provides: get_input_embeddings().weight [V,H], lm_head (rank-64, cheap),
    forward(...) with per-layer hidden states h_i = embed * (1 + i/64), and a
    real DynamicCache it appends to.

    Logits run in 'echo' mode: the target's greedy argmax agrees with the
    block's next token for the first K positions of each verify pass, then
    deliberately mismatches — K cycles through ``pattern`` — so the trace
    exercises varied acceptance lengths (ctx appends of 1..8) deterministically.
    Semantic quality is irrelevant here; only cache/position/mask realism is.
    """

    PATTERN = [7, 3, 1, 5, 2, 0, 6, 4]

    def __init__(self, vocab, hidden, num_layers=64):
        super().__init__()
        self.config = Qwen3Config(vocab_size=vocab, hidden_size=hidden,
                                  num_hidden_layers=num_layers,
                                  num_key_value_heads=8, head_dim=128,
                                  intermediate_size=hidden * 2)
        self.device = torch.device("cpu")
        emb = torch.empty(vocab, hidden)
        torch.manual_seed(1)
        emb.normal_(0, 0.02)
        self.embed_tokens = torch.nn.Embedding(vocab, hidden, _weight=emb)
        self.lm_head = _RankHead(hidden, vocab)
        self._verify_calls = 0
        self._cache = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, position_ids=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, output_hidden_states=False,
                **kw):
        L = input_ids.shape[1]
        h0 = self.embed_tokens(input_ids)
        out = {}
        if output_hidden_states:
            out["hidden_states"] = tuple(
                h0 * (1.0 + i / 64.0) for i in range(self.config.num_hidden_layers + 1))
        # echo-mode logits: argmax[t] = ids[t+1] for t < K (accept),
        # then a guaranteed mismatch (ids[t+1]+7) — acceptance length = K.
        K = self.PATTERN[self._verify_calls % len(self.PATTERN)]
        is_verify = L <= 8 and not logits_to_keep
        if is_verify:
            self._verify_calls += 1
        logits = torch.zeros(1, L, self.config.vocab_size)
        for t in range(L - 1):
            agree = (not is_verify) or t < K
            tok = int(input_ids[0, t + 1]) if agree \
                else (int(input_ids[0, t + 1]) + 7) % self.config.vocab_size
            logits[0, t, tok] = 50.0
        logits[0, L - 1, 123] = 50.0                      # bonus slot
        k = int(logits_to_keep) if logits_to_keep else 0
        out["logits"] = logits[:, -k:] if k else logits
        # honest cache bookkeeping (values are zeros; lengths are real)
        if past_key_values is not None:
            zk = torch.zeros(1, L, 8 * 128)
            zv = torch.zeros(1, L, 8 * 128)
            for li in range(self.config.num_hidden_layers):
                past_key_values.update(zk, zv, li, {})
        return SimpleNamespace(**out)


class _RankHead(torch.nn.Module):
    def __init__(self, h, v):
        super().__init__()
        torch.manual_seed(2)
        self.a = torch.nn.Parameter(torch.randn(h, 64) * 0.05)
        self.b = torch.nn.Parameter(torch.randn(64, v) * 0.05)

    def forward(self, h):
        return (h @ self.a) @ self.b


# ---------------------------------------------------------------------------
# Mask instrumentation: wrap dflash.model._attention_mask (referenced by the
# attention module as a module global) to record shape + visibility stats.
# ---------------------------------------------------------------------------

mask_log = []

def _instrumented_attention_mask(query, key, *, is_causal, sliding_window):
    m = _attention_mask(query, key, is_causal=is_causal,
                        sliding_window=sliding_window)
    ql, kl = m.shape[-2], m.shape[-1]
    vis = m[0, 0]
    per_q = vis.sum(-1)
    # key i is at absolute position (kl - ql) + i  for the *new* keys only if
    # the cache was empty; caller supplies cache_before via globals for abs
    first_q, last_q = per_q[0].item(), per_q[-1].item()
    # first noise query row: min/max visible key index
    row = vis[0].nonzero()
    mask_log.append({
        "shape": [ql, kl], "is_causal": bool(is_causal),
        "sliding_window": sliding_window,
        "q0_visible": int(first_q), "qlast_visible": int(last_q),
        "q0_min_key": int(row.min()) if row.numel() else -1,
        "q0_max_key": int(row.max()) if row.numel() else -1,
        "vis_frac": float(vis.float().mean()),
    })
    return m


DM._attention_mask = _instrumented_attention_mask


# ---------------------------------------------------------------------------
# The cycle loop — transcribed from dflash.model.dflash_generate with
# instrumentation inline. Semantics identical; helpers called verbatim.
# ---------------------------------------------------------------------------

def traced_generate(model, target, input_ids, max_new_tokens, stop_token_ids,
                    temperature, top_p=1.0, top_k=0, log=print):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size

    output_ids = torch.full((1, max_length + 1), model.mask_token_id,
                            dtype=torch.long)
    position_ids = torch.arange(output_ids.shape[1]).unsqueeze(0)
    past_target = _make_cache(target.config)
    past_draft = _make_cache(model.config)

    # ---- prefill (target) ----
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
    log(f"prefill: prompt={num_input_tokens} target_hidden={tuple(target_hidden.shape)}")

    acceptance_lengths = []
    start = num_input_tokens
    stop_tokens = (torch.tensor(stop_token_ids) if stop_token_ids else None)
    stopped = (stop_tokens is not None
               and torch.isin(output_ids[:, start], stop_tokens).any())
    trace = []
    cycle = 0

    while start + 1 < max_length and not stopped:
        verify_size = min(block_size, max_length - start)
        block_output_ids = output_ids[:, start:start + verify_size].clone()
        block_position_ids = position_ids[:, start:start + verify_size]
        ctx_len = target_hidden.shape[1]

        cache_before = past_draft.get_seq_length()
        invariant_ok = cache_before == start - ctx_len

        if verify_size > 1:
            noise_embedding = _raw_input_embeddings(
                target, block_output_ids,
                float(DM._draft_value(model.config, "input_embedding_scale", 1.0)))
            draft_pos = position_ids[:, start - ctx_len:start + verify_size]
            mask_log.clear()
            draft_hidden = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_pos,
                past_key_values=past_draft,
                use_cache=True)[:, 1 - verify_size:, :]
            _crop_to(past_draft, start)
            if isinstance(model, DFlash2DraftModel):
                draft_tokens, draft_indices, draft_probs = model.propose(
                    draft_hidden, block_output_ids[:, 0],
                    _output_head(target), temperature)
                block_output_ids[:, 1:] = draft_tokens

        rec = {
            "cycle": cycle, "start": start, "verify_size": verify_size,
            "ctx_len": ctx_len,
            "ctx_pos": [start - ctx_len, start],
            "noise_pos": [start, start + verify_size],
            "draft_cache_before": cache_before,
            "invariant_cache_eq_start_minus_ctx": invariant_ok,
            "draft_cache_after_crop": past_draft.get_seq_length(),
            "mask": dict(mask_log[-1]) if mask_log else None,
        }

        # ---- verify (target) ----
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
            stop_idx = torch.isin(
                output_ids[0, start + 1:start + produced + 1],
                stop_tokens).nonzero(as_tuple=True)[0]
            if stop_idx.numel() > 0:
                produced = stop_idx[0].item() + 1
                stopped = True
        start += produced
        _crop_to(past_target, start)
        acceptance_lengths.append(produced)

        rec.update({"accepted": acceptance_length, "produced": produced,
                    "target_cache_after_crop": past_target.get_seq_length()})
        if verify_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, model.target_layer_ids)[:, :produced, :]
        rec["next_ctx_len"] = target_hidden.shape[1]
        rec["next_ctx_pos"] = [start - target_hidden.shape[1], start]
        trace.append(rec)
        cycle += 1

    return output_ids[:, :min(start + 1, max_length)], acceptance_lengths, trace


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["short", "shortT", "long", "all"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    modes = ["short", "shortT", "long"] if args.mode == "all" else [args.mode]

    torch.manual_seed(0)
    print("== loading DFlash2 draft (fp32 CPU) ...")
    model = DFlash2DraftModel.from_pretrained(CKPT, torch_dtype=torch.bfloat16)
    model = model.float().eval()
    V = model.config.vocab_size
    print("== building stub target ...")
    target = StubTarget(V, model.config.hidden_size)

    prompts = {"short": 32, "shortT": 32, "long": 2100}
    temps = {"short": 0.0, "shortT": 1.0, "long": 0.0}

    for mode in modes:
        n = prompts[mode]
        torch.manual_seed(0)
        ids = torch.randint(1000, 2000, (1, n))
        print(f"== mode {mode}: prompt {n}, T={temps[mode]}")
        out, acc, trace = traced_generate(
            model, target, ids, max_new_tokens=48, stop_token_ids=None,
            temperature=temps[mode])
        path = os.path.join(OUT_DIR, f"cache_trace_{mode}.jsonl")
        with open(path, "w") as f:
            for rec in trace:
                f.write(json.dumps(rec) + "\n")
        inv = all(r["invariant_cache_eq_start_minus_ctx"] for r in trace)
        masks = [r["mask"] for r in trace if r["mask"]]
        print(f"   cycles={len(trace)} produced={acc}")
        print(f"   invariant draft_cache==start-ctx held every cycle: {inv}")
        if masks:
            m0 = masks[0]
            print(f"   mask[0]: shape={m0['shape']} q0_visible={m0['q0_visible']} "
                  f"vis_frac={m0['vis_frac']:.3f} window={m0['sliding_window']}")
        print(f"   wrote {path} ({len(trace)} records)")


if __name__ == "__main__":
    main()
