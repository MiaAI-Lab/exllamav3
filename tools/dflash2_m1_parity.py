#!/usr/bin/env python3
"""M1 numerics-parity gate: DFlash2Model (exllamav3) vs dflash reference,
using REAL per-cycle fixtures from M0.4
(test_models/dspark-evals/dflash2/parity_fixtures/p0_greedy_c*.pt — greedy/T=0,
so the reference selector walk is the same greedy walk exl3 implements).

For every cycle: rebuild the exact per-cycle inputs the reference consumed
(ctx taps from ctx_in, noise embedding from the target embedding table rows,
anchor from block_ids), drive update_kv_from_target + forward +
sample_from_state through the exl3 draft, and compare against the saved
reference outputs:

  [1] draft hidden state (post-norm, rows 1..7) — per-row rel-Frobenius
  [2] selector path tokens (greedy walk) — match count
  [3] top-16 candidate sets (logits topk) — overlap

Gate: all cycles max-row-err < 6% & cosine > 0.9995 (bf16 ref vs fp16 exl3;
DSpark measured ~1.9-4.7% pure-dtype drift, 0.2% like-for-like), tokens >= 6/7.

Run: .venv/bin/python tools/dflash2_m1_parity.py [--cycles N]
"""

import argparse, glob, os, sys
import torch
import torch.nn.functional as F

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from safetensors import safe_open

from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.cache import Cache
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.modules.module import Module


class _StubEmbed(Module):
    """Target embed stand-in; returns a precomputed noise embedding."""
    def __init__(self, config, noise_embedding):
        super().__init__(config, "stub_embed", None)
        self.noise_embedding = noise_embedding
        self.device = noise_embedding.device

    def optimizer_targets(self):
        raise NotImplementedError()

    def forward(self, x, params, out_dtype = None):
        return self.noise_embedding


class _StubLMHead(Module):
    """Target lm_head stand-in; F.linear with the fixture's weight."""
    def __init__(self, config, weight):
        super().__init__(config, "stub_lm_head", None)
        self.weight = weight
        self.device = weight.device

    def optimizer_targets(self):
        raise NotImplementedError()

    def forward(self, x, params, out_dtype = None):
        return F.linear(x, self.weight)


class _FakeTarget:
    def __init__(self, stub_embed, stub_lm_head, vocab_size):
        self.modules = [stub_embed, stub_lm_head]
        self.logit_layer_idx = 1
        self.loaded_tp = False
        self.config = type("C", (), {"vocab_size": vocab_size})()


def load_target_rows(ids, lm_head_too):
    """Read embedding rows (and the full lm_head) from HF safetensors."""
    idx = os.path.join(repo_root, "test_models/sources/Qwen3.8-27B",
                       "model.safetensors.index.json")
    import json
    index = json.load(open(idx))["weight_map"]
    dev = torch.device("cuda:0")

    needed = {i: "model.language_model.embed_tokens.weight" for i in ids}
    emb_rows = {}
    opened = {}
    for tid, tensor_name in needed.items():
        shard = index[tensor_name]
        if shard not in opened:
            path = os.path.join(os.path.dirname(idx), shard)
            opened[shard] = safe_open(path, framework = "pt")
        emb_rows[tid] = opened[shard].get_slice(tensor_name)[tid].to(
            torch.bfloat16).to(dev)

    lm_w = None
    if lm_head_too:
        shards = sorted(set(index[t] for t in index if "lm_head" in t))
        parts = []
        from safetensors.torch import load_file
        for s in shards:
            sd = load_file(os.path.join(os.path.dirname(idx), s))
            for k, v in sd.items():
                if "lm_head" in k:
                    parts.append(v)
        lm_w = torch.cat(parts, dim = 0).to(dev).half()  # [V, H] fp16
    return emb_rows, lm_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-dir", default = "test_models/sources/Qwen3.8-27B-DFlash2")
    ap.add_argument("--fixtures", default =
                    "test_models/dspark-evals/dflash2/parity_fixtures/p0_greedy_c*.pt")
    ap.add_argument("--cycles", type = int, default = 100)
    args = ap.parse_args()

    device = torch.device("cuda:0")

    config = Config.from_directory(os.path.join(repo_root, args.draft_dir))
    from exllamav3.architecture.dflash2 import DFlash2Config, DFlash2Model
    assert isinstance(config, DFlash2Config)
    model = Model.from_config(config)

    files = sorted(glob.glob(os.path.join(repo_root, args.fixtures)))[:args.cycles]
    fix = [torch.load(f, map_location = "cpu", weights_only = False) for f in files]
    # Final-cycle fixtures can carry a truncated block (verify_size < block_size at
    # the length cap); the exl3 draft always emits a full block, so skip those.
    fix = [f for f in fix if f["draft_hidden"].shape[1] == 7]
    print(f"loaded {len(fix)} cycles")

    mask_id = config.mask_token_id
    anchors = sorted({int(f["block_ids"][0, 0]) for f in fix})
    emb_rows, lm_w = load_target_rows(anchors + [mask_id], lm_head_too = True)
    print(f"embedding rows for {len(anchors)} anchors + mask; lm_head {tuple(lm_w.shape)}")

    fake_target_holder = {}
    draft_cache = Cache(model, max_num_tokens = 512, layer_type = CacheLayer_fp16)
    model.load(device)
    print("draft loaded")

    ok_all = True
    tok_total = tok_match = top16_total = top16_match = 0
    max_err_overall, min_cos_overall = 0.0, 1.0
    per_cycle = []

    # The reference draft cache accumulates K/V over the WHOLE generation:
    # each cycle appends its newly verified ctx rows (and crops only the tail
    # noise). Reproduce that by chaining fixtures: cycle t's ctx_in lands at
    # [pos, pos + L); noise at [pos, pos + 8) transiently; pos += produced.
    block_table = torch.zeros((1, 1), dtype = torch.int32, device = device)
    pos = 0
    for f in fix:
        L = f["ctx_in"].shape[1]
        H = config.hidden_size
        ntaps = len(config.target_layer_ids)
        states = [f["ctx_in"][:, :, i * H:(i + 1) * H] for i in range(ntaps)]
        anchor = int(f["block_ids"][0, 0])

        # Write this cycle's ctx rows at [pos, pos + L)
        model.update_kv_from_target(
            target_hidden = [s.to(device, torch.half) for s in states],
            cache = draft_cache,
            params = {"block_table": block_table,
                      "cache_seqlens": torch.tensor([pos], dtype = torch.int32,
                                                     device = device)},
        )
        pos_end = pos + L

        noise = torch.stack([emb_rows[anchor]] + [emb_rows[mask_id]] * 7,
                            dim = 0).unsqueeze(0)                  # [1, 8, H]
        fake = _FakeTarget(
            _StubEmbed(config, noise.to(device, torch.half)),
            _StubLMHead(config, lm_w),
            config.vocab_size)
        model.attach_to(fake)
        fake_target_holder["fake"] = fake                          # keep alive

        params = {"attn_mode": "flash_attn", "block_table": block_table,
                  "cache": draft_cache,
                  "cache_seqlens": torch.tensor([pos_end], dtype = torch.int32,
                                                 device = device)}
        state = model.forward(input_ids = torch.tensor([[anchor]]), params = params)
        new_ids = model.sample_from_state(state, params)
        pos = pos_end    # next cycle's ctx rows continue after this window

        # -- [1] hidden parity (rows 1..7)
        ref_f = f["draft_hidden"].to(device).float()
        got_f = state[:, 1:, :].float()
        row_errs = [((got_f[0, j] - ref_f[0, j]).norm() /
                     ref_f[0, j].norm()).item() for j in range(7)]
        max_row_err = max(row_errs)
        cos = F.cosine_similarity(got_f.flatten(), ref_f.flatten(), dim = 0).item()

        # -- [2] token parity
        got_tok = new_ids[0, 1:].cpu()
        ref_tok = f["tokens"][0]
        m = int((got_tok == ref_tok).sum())

        # -- [3] top-16 candidate overlap from the same logits
        with torch.inference_mode():
            lg = F.linear(state[:, 1:, :].half(), lm_w).float()
        got_top = torch.topk(lg, 16, dim = -1).indices[0]
        ref_top = f["indices"][0]
        ov = int(torch.tensor([[len(set(a.tolist()) & set(b.tolist()))
                                for a, b in zip(got_top, ref_top)]]).sum())
        top16_total += 7 * 16
        top16_match += ov

        tok_total += 7; tok_match += m
        max_err_overall = max(max_err_overall, max_row_err)
        min_cos_overall = min(min_cos_overall, cos)
        ok = max_row_err < 0.06 and cos > 0.9995
        ok_all &= ok
        per_cycle.append(dict(cycle = int(f["cycle"]), L = L, anchor = anchor,
                              err = round(max_row_err, 5), cos = round(cos, 6),
                              tok = f"{m}/7", top16 = f"{ov}/{7 * 16}"))
        flag = "" if ok else "  <-- ERR/COS FAIL"
        if not ok or m < 6 or int(f["cycle"]) < 3:
            print(f" cyc {f['cycle']:2d} L={L:3d} err={max_row_err:.4f} "
                  f"cos={cos:.6f} tok={m}/7 top16={ov}/{7 * 16}{flag}")

    print()
    print(f"[1] hidden: max per-row rel-Frobenius over all cycles = {max_err_overall:.4f}")
    print(f"    min cosine = {min_cos_overall:.6f}")
    print(f"[2] tokens: {tok_match}/{tok_total} match")
    print(f"[3] top-16 candidate overlap: {top16_match}/{top16_total} "
          f"({100.0 * top16_match / top16_total:.1f}%)")
    gate = (ok_all and min_cos_overall > 0.9995 and
            tok_match >= tok_total - tok_total // 20)
    print()
    print("M1 PARITY GATE:", "PASS" if gate else "FAIL")
    import json as _json
    _json.dump(per_cycle, open(os.path.join(
        repo_root, "test_models/dspark-evals/dflash2/m1_parity_result.json"), "w"),
        indent = 1)
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
