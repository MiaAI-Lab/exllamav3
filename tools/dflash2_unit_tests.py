#!/usr/bin/env python3
"""M0.2 — DFlash2 reference CPU unit tests (dspark-imp.md §M0.2).

Server-safe (no GPU, no target model). Instantiates the *reference*
DFlash2DraftModel from the downloaded checkpoint and verifies:

  T1  checkpoint actually loads (codebooks not silently random-init)
  T2  forward: shapes + draft cache append/crop semantics
  T3  extract_context_feature: layer_id+1 offset, concat dim
  T4  GroupedDynamicCausalConv: causality (future cannot leak to past)
  T5  CandidateSelector: shapes, T=0 argmax walk, T>0 q-rows are
      distributions over the 16 candidates, path tokens ⊆ candidates
  T6  propose with a stub output head (end-to-end, real weights)
  T7  _rejection_sample: analytical cases (accept-all, reject-at-0,
      duplicate candidates, draft_indices=None branch, mid-reject residual)
  T8  _attention_mask: bilateral window hand case vs causal contrast

Run: .venv/bin/python tools/dflash2_unit_tests.py
"""

import sys, os, math
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch
from types import SimpleNamespace
from transformers import DynamicCache
from safetensors import safe_open

from dflash.model import (
    DFlash2DraftModel,
    CandidateSelector,
    GroupedDynamicCausalConv,
    extract_context_feature,
    _rejection_sample,
    _attention_mask,
    _crop_to,
    _make_cache,
    _draft_value,
)

CKPT = os.path.join(repo_root, "test_models/sources/Qwen3.8-27B-DFlash2")
DEVICE = "cpu"

PASS, FAIL = 0, 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  [PASS] {name}")
    else: FAIL += 1; print(f"  [FAIL] {name} {detail}")

torch.manual_seed(0)

print("== loading DFlash2DraftModel (bf16 -> fp32 on CPU) ...")
model = DFlash2DraftModel.from_pretrained(CKPT, torch_dtype=torch.bfloat16)
model = model.float().to(DEVICE).eval()
cfg = model.config
H = cfg.hidden_size
BLOCK = int(_draft_value(cfg, "block_size"))
MASK_ID = int(_draft_value(cfg, "mask_token_id"))
CTX_DIM = 5 * H

# ---------------------------------------------------------------- T1
print("== T1: checkpoint load verification")
with safe_open(os.path.join(CKPT, "model.safetensors"), framework="pt") as f:
    pcb = f.get_slice("candidate_selector.predecessor_codebook")[:5]
    emb = f.get_slice("layers.0.self_attn.q_proj.weight")[:3, :4]
check("codebook weights match safetensors",
      torch.equal(model.candidate_selector.predecessor_codebook.weight[:5], pcb.float()))
check("attn weights match safetensors",
      torch.equal(model.layers[0].self_attn.q_proj.weight[:3, :4], emb.float()))
check("block_size/mask id", BLOCK == 8 and MASK_ID == 248070, f"{BLOCK} {MASK_ID}")
check("target_layer_ids", list(model.target_layer_ids) == [5, 19, 33, 47, 61])

# ---------------------------------------------------------------- T2
print("== T2: forward shapes + cache semantics")
L = 12
target_hidden = torch.randn(1, L, CTX_DIM)
noise_embedding = torch.randn(1, BLOCK, H)
pos = torch.arange(L + BLOCK).unsqueeze(0)
cache = _make_cache(cfg)
with torch.inference_mode():
    out = model(position_ids=pos, noise_embedding=noise_embedding,
                target_hidden=target_hidden, past_key_values=cache, use_cache=True)
check("forward output shape [1, block, H]",
      out.shape == (1, BLOCK, H), f"got {tuple(out.shape)}")
check("cache grew by L+block", cache.get_seq_length() == L + BLOCK,
      f"got {cache.get_seq_length()}")
_crop_to(cache, L)
check("crop_to(L) -> seq_len L", cache.get_seq_length() == L)
# second cycle appends exactly the new ctx positions (L2) + block, then crops
L2 = 3
target_hidden2 = torch.randn(1, L2, CTX_DIM)
pos2 = torch.arange(L, L + L2 + BLOCK).unsqueeze(0)
with torch.inference_mode():
    out2 = model(position_ids=pos2, noise_embedding=noise_embedding,
                 target_hidden=target_hidden2, past_key_values=cache, use_cache=True)
check("cycle-2 cache = L + L2 + block", cache.get_seq_length() == L + L2 + BLOCK,
      f"got {cache.get_seq_length()}")

# ---------------------------------------------------------------- T3
print("== T3: extract_context_feature offset")
hs = [torch.full((1, 4, H), float(i)) for i in range(65)]
feat = extract_context_feature(hs, [5, 19])
check("shape [.., len(ids)*H]", feat.shape == (1, 4, 2 * H), f"got {tuple(feat.shape)}")
check("offset +1 (ids 5,19 -> states 6,20)",
      feat[0, 0, 0].item() == 6.0 and feat[0, 0, H].item() == 20.0)

# ---------------------------------------------------------------- T4
print("== T4: dynconv causality")
conv = GroupedDynamicCausalConv(H, 2, 16).eval()
x = torch.randn(1, BLOCK, H)
with torch.inference_mode():
    pre, dyn = conv.prepare(x)
    y = conv.finish(pre, dyn)
    x2 = x.clone(); x2[:, -1] += 10.0          # perturb the future
    pre2, dyn2 = conv.prepare(x2)
    y2 = conv.finish(pre2, dyn2)
check("prepare output causal", torch.equal(y[:, :-1], y2[:, :-1]))
check("finish output causal", torch.equal(y[:, :-1], y2[:, :-1]))
check("last position changed", not torch.equal(y[:, -1], y2[:, -1]))

# ---------------------------------------------------------------- T5
print("== T5: CandidateSelector mechanics")
sel_cfg = SimpleNamespace(dflash_config=dict(selector_rank=256, selector_top_k=16),
                          vocab_size=1000, hidden_size=64)
sel = CandidateSelector(sel_cfg).eval()
with torch.no_grad():
    sel.hidden_projection.weight.zero_()        # gate = 0 -> S_t = U_t
hidden = torch.randn(1, 7, 64)
logits = torch.randn(1, 7, 1000)
with torch.inference_mode():
    path, cands, q = sel.select(hidden, logits, torch.tensor([7]), 0.0)
    top1 = logits.argmax(-1)
    in_top16 = torch.isin(path, cands).all()
check("T=0 shapes", path.shape == (1, 7) and cands.shape == (1, 7, 16) and q is None)
check("T=0 path = per-position top-1 (zeroed gate)", torch.equal(path[0], top1[0]))
check("path tokens ⊆ candidates", bool(in_top16))
with torch.inference_mode():
    path_s, cands_s, q_s = sel.select(hidden, logits, torch.tensor([7]), 1.0)
check("T>0 q rows shape", q_s.shape == (1, 7, 16))
check("T>0 q rows are distributions",
      bool((q_s.sum(-1) - 1).abs().max() < 1e-5))
check("T>0 path ⊆ candidates", torch.isin(path_s, cands_s).all().item())

# ---------------------------------------------------------------- T6
print("== T6: propose end-to-end (real weights, stub head)")
class StubHead(torch.nn.Module):
    """rank-64 head: [.., H] @ [H,64] @ [64, vocab] — cheap but full-width."""
    def __init__(s, h, v):
        super().__init__()
        s.a = torch.nn.Parameter(torch.randn(h, 64))
        s.b = torch.nn.Parameter(torch.randn(64, v))
    def forward(s, h):
        return (h @ s.a) @ s.b
head = StubHead(H, cfg.vocab_size)
h7 = out[:, 1:, :]                              # draft positions 1..7
with torch.inference_mode():
    toks, idx, probs = model.propose(h7, torch.tensor([123]), head, 0.0)
    toks_s, idx_s, probs_s = model.propose(h7, torch.tensor([123]), head, 1.0)
check("T=0 tokens/idx shapes", toks.shape == (1, 7) and idx.shape == (1, 7, 16))
check("T=0 probs None", probs is None)
check("T>0 probs are dists over 16",
      probs_s.shape == (1, 7, 16) and bool((probs_s.sum(-1) - 1).abs().max() < 1e-4))
check("tokens ⊆ candidate idx", bool((idx_s == toks_s[..., None]).sum(-1).all()))

# ---------------------------------------------------------------- T7
print("== T7: _rejection_sample analytical")
V, G = 50, 7
def onehot(v, idx):
    t = torch.zeros(1, v); t[0, idx] = 1.0; return t

# a) perfect match: q onehot on draft tokens, p same -> accept all + bonus
dt = torch.randint(0, V, (1, G))
p = torch.zeros(1, G + 1, V); p[0, torch.arange(G), dt[0]] = 1.0
p[0, G, 11] = 1.0                               # bonus distribution
di = torch.randint(0, 16, (1, G, 16)); dp = torch.zeros(1, G, 16); dp[..., 0] = 1.0
di[..., 0] = dt                                  # candidate 0 == draft token
acc, bonus = _rejection_sample(dt, p, dp, di)
check("accept-all case", acc == G and bonus.item() == 11, f"acc={acc} bonus={bonus.item()}")

# b) p(x_0) = 0 -> reject at 0, resample = argmax(p - q) at position 0
p2 = torch.zeros(1, G + 1, V); p2[0, 0, 33] = 1.0; p2[0, G, 11] = 1.0
di2 = torch.zeros(1, G, 16, dtype=torch.long); dp2 = torch.zeros(1, G, 16); dp2[..., 0] = 1.0
di2[..., 0] = dt; dp2[..., 0] = 1.0              # q onehot on dt (=draft token)
acc2, bonus2 = _rejection_sample(dt, p2, dp2, di2)
check("reject-at-0 case", acc2 == 0 and bonus2.item() == 33,
      f"acc={acc2} bonus={bonus2.item()}")

# c) duplicate candidates: two slots equal draft token -> q = sum
p3 = torch.zeros(1, G + 1, V)
for i in range(G): p3[0, i, dt[0, i]] = 1.0   # p onehot on draft tokens
p3[0, G, dt[0, -1]] = 1.0                      # bonus position (unused for acc)
di3 = torch.randint(0, 16, (1, G, 16)); dp3 = torch.zeros(1, G, 16)
di3[..., 0] = dt; dp3[..., 0] = 0.6
di3[..., 1] = dt; dp3[..., 1] = 0.4              # same token in 2 slots -> q=1.0
acc3, _ = _rejection_sample(dt, p3, dp3, di3)
check("duplicate-candidate q-sum", acc3 == G, f"acc={acc3}")

# d) draft_indices=None branch: q = gather
dp4 = torch.zeros(1, G, V); dp4[0, torch.arange(G), dt[0]] = 1.0
acc4, bonus4 = _rejection_sample(dt, p, dp4, None)
check("no-indices branch accept-all", acc4 == G and bonus4.item() == 11)

# e) mid-reject: q=1 on token a, p=0.5 on a and 0.5 on b at position 1 ->
#    rand*1 < 0.5 fails ~50%; force it by p(a)=0 at position 1 (reject there),
#    residual = norm(p - q) must put mass on b
dt5 = torch.tensor([[5, 6, 7, 8, 9, 10, 11]])
p5 = torch.zeros(1, G + 1, V)
p5[0, 0, dt5[0, 0]] = 1.0                        # position 0: match
p5[0, 1, 42] = 1.0                               # position 1: p(token 6)=0 -> reject
di5 = torch.zeros(1, G, 16, dtype=torch.long); dp5 = torch.zeros(1, G, 16)
di5[..., 0] = dt5; dp5[..., 0] = 1.0             # q onehot on draft tokens
acc5, bonus5 = _rejection_sample(dt5, p5, dp5, di5)
check("mid-reject residual", acc5 == 1 and bonus5.item() == 42,
      f"acc={acc5} bonus={bonus5.item()}")

# ---------------------------------------------------------------- T8
print("== T8: bilateral attention mask (hand case)")
q = torch.zeros(1, 4, 3, 16)   # q_len 3
k = torch.zeros(1, 4, 10, 16)  # ctx 7 + noise 3
m = _attention_mask(q, k, is_causal=False, sliding_window=4)
check("mask shape", m.shape == (1, 1, 3, 10))
vis = m[0, 0]
exp = torch.zeros(3, 10, dtype=torch.bool)
for qi, qp in enumerate([7, 8, 9]):
    for ki, kp in enumerate(range(10)):
        exp[qi, ki] = abs(qp - kp) < 4
check("bilateral window pattern", torch.equal(vis, exp))
check("noise sees noise (q=7 sees k=8,9)", bool(vis[0, 8] and vis[0, 9]))
mc = _attention_mask(q, k, is_causal=True, sliding_window=4)
check("causal contrast: q=7 cannot see k=8,9", not bool(mc[0, 0, 0, 8] or mc[0, 0, 0, 9]))

print(f"\n== {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
