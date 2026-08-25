#!/usr/bin/env python
"""
M1 numerics-parity gate: DSparkDraftModel (exllamav3) vs the SpecForge reference.

Compares, on a fixed single-block case with REAL target-model inputs
(reference/dspark_m1_fixture.pt, produced by tools in M0):
  1. draft hidden state (post-norm, all block rows)   -- core parity
  2. markov-corrected sampled drafts
  3. confidence head output (sigmoid)

Usage:
  python tools/dspark_m1_parity.py [--draft-dir DIR] [--fixture FILE] [--threshold T]
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.cache import Cache
from exllamav3.cache.fp16 import CacheLayer_fp16
from exllamav3.modules.module import Module


class _StubEmbed(Module):
    """Stands in for the target's embed_tokens; returns the fixture's noise embedding."""
    def __init__(self, config, noise_embedding):
        super().__init__(config, "stub_embed", None)
        self.noise_embedding = noise_embedding
        self.device = noise_embedding.device

    def optimizer_targets(self):
        raise NotImplementedError()

    def forward(self, x, params, out_dtype = None):
        return self.noise_embedding


class _StubLMHead(Module):
    """Stands in for the target's lm_head; F.linear with the fixture's weight."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-dir", default = "test_models/sources/Qwen3.8-27B-DSpark")
    parser.add_argument("--fixture", default = "reference/dspark_m1_fixture.pt")
    parser.add_argument("--threshold", type = float, default = 0.05,
                        help = "max relative error on draft hidden state")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    fix = torch.load(args.fixture, map_location = "cpu", weights_only = False)

    # -- Build & load the draft model through the normal exl3 paths
    config = Config.from_directory(args.draft_dir)
    from exllamav3.architecture.dspark import DSparkDraftConfig, DSparkDraftModel
    assert isinstance(config, DSparkDraftConfig), f"got {type(config)}"
    assert config.target_layer_ids == [4, 16, 28, 40, 52], config.target_layer_ids
    assert config.block_size == 7

    model = Model.from_config(config)
    assert isinstance(model, DSparkDraftModel)
    assert model.caps["default_draft_size"] == 7

    draft_cache = Cache(model, max_num_tokens = 256, layer_type = CacheLayer_fp16)
    model.load(device)
    print(" -- draft model loaded")

    # -- Attach stub target providing the fixture's embedding / lm_head
    noise_embedding = fix["noise_embedding"].to(device, torch.half)
    lm_head_w = fix["lm_head_weight"].to(device, torch.half)
    fake_target = _FakeTarget(
        _StubEmbed(config, noise_embedding),
        _StubLMHead(config, lm_head_w),
        fix["lm_head_weight"].shape[0],
    )
    model.attach_to(fake_target)

    # -- Write draft context KV from the fixture's tap states
    anchor = fix["anchor"]                                     # 10
    taps = fix["target_hidden_taps"]                           # (1, anchor, 5*hidden)
    n_taps = len(config.target_layer_ids)
    hidden = taps.shape[-1] // n_taps
    target_hidden = [taps[..., i * hidden:(i + 1) * hidden].to(device, torch.half)
                     for i in range(n_taps)]

    block_table = torch.zeros((1, 1), dtype = torch.int32, device = device)  # page 0
    cache_seqlens_0 = torch.zeros((1,), dtype = torch.int32, device = device)

    model.update_kv_from_target(
        target_hidden = target_hidden,
        cache = draft_cache,
        params = {"block_table": block_table, "cache_seqlens": cache_seqlens_0},
    )
    print(f" -- wrote {anchor} context rows into draft cache")

    # -- Draft forward
    seed_ids = fix["block_ids"][:, :1].cpu()                   # (1, 1) anchor token
    cache_seqlens = torch.tensor([anchor], dtype = torch.int32, device = device)
    params = {
        "attn_mode": "flash_attn",
        "block_table": block_table,
        "cache": draft_cache,
        "cache_seqlens": cache_seqlens,
    }
    state = model.forward(input_ids = seed_ids, params = params)   # (1, 7, hidden) fp16
    print(f" -- draft forward: {tuple(state.shape)}")

    # -- 1. Draft hidden parity
    # (bf16 reference vs fp16 exl3: pure dtype drift alone is ~1.9% rel-Frobenius,
    #  measured by running the reference in fp16; exl3 vs reference-fp16 is ~0.2%)
    ref = fix["draft_hidden_out"].to(device).float()
    got = state.float()
    row_errs = [((got[0, j] - ref[0, j]).norm() / ref[0, j].norm()).item() for j in range(got.shape[1])]
    max_row_err = max(row_errs)
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim = 0).item()
    print(f" [1] draft hidden: max per-row rel-Frobenius = {max_row_err:.4f}, cosine = {cos:.6f}")
    ok1 = max_row_err < 0.025 and cos > 0.9995

    # -- 2. Sampled drafts parity
    new_ids = model.sample_from_state(state, params)           # (1, 8) [seed, d1..d7]
    ref_drafts = fix["sampled_drafts"].to(device)
    got_drafts = new_ids[:, 1:]
    match = (got_drafts == ref_drafts).sum().item()
    print(f" [2] sampled drafts: {match}/7 match reference")
    print(f"     ref: {ref_drafts[0].tolist()}")
    print(f"     got: {got_drafts[0].tolist()}")
    ok2 = match == 7

    # -- 3. Confidence parity (recompute exactly as sample_from_state does)
    dev = model.confidence.device
    out = new_ids.to(dev)
    embs = torch.stack([F.embedding(out[:, i], model._markov_w1_dev).half() for i in range(7)], dim = 1)
    conf = model.confidence.forward(torch.cat((state.to(dev).half(), embs), dim = -1), params)
    cs = torch.sigmoid(conf.float().squeeze(-1))[0]
    ref_conf = torch.sigmoid(fix["confidence"].to(device).float())[0]
    conf_err = (cs - ref_conf).abs().max().item()
    print(f" [3] confidence: max abs err (sigmoid) = {conf_err:.4f}")
    print(f"     ref: {[f'{v:.3f}' for v in ref_conf.tolist()]}")
    print(f"     got: {[f'{v:.3f}' for v in cs.tolist()]}")
    ok3 = conf_err < 0.05

    # -- 4. draft_confidence_len sanity
    conf_len = params.get("draft_confidence_len")
    print(f" [4] draft_confidence_len = {conf_len} (threshold {model.draft_conf_threshold})")

    print()
    if ok1 and ok2 and ok3:
        print("M1 PARITY GATE: PASS")
        return 0
    print(f"M1 PARITY GATE: FAIL (hidden={ok1} drafts={ok2} conf={ok3})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
