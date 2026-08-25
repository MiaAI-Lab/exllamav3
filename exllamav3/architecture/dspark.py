from __future__ import annotations
from typing_extensions import override
import os
import torch
import torch.nn.functional as F
import weakref

from .dflash import DFlashConfig, DFlashModel
from ..modules import Module, Embedding, Linear
from ..modules.arch_specific.dspark import to_dev

# DSpark draft model (DFlash backbone + Markov/confidence heads), e.g.
# RadixArk/Qwen3.8-27B-DSpark. Trained with SpecForge, served upstream with SGLang
# (--speculative-algorithm DSPARK).
#
# Semantics (verified against SpecForge training code and the actual weights; see
# M0_DSPARK_SPIKE.md):
#   - Block input = [anchor, mask x (block_size-1)]; EVERY row produces a draft:
#     row j predicts the token at anchor+j+1 (next-token alignment, unlike plain
#     z-lab DFlash where rows predict their own position and the anchor row's
#     output is discarded). Drafts = block_size, verify width = block_size + 1.
#   - Taps are the raw residual-stream OUTPUT of target layer i (SpecForge reads
#     hidden_states[i + 1] under the HF convention) => tap_shift 0.
#   - Context K/V for the draft cache are derived per layer from the projected tap
#     stream  hidden_norm(fc(concat(taps)))  (update_kv_from_target).
#   - Markov head: sequential bigram bias  logits[j] += w2(w1(prev_j))  with
#     prev_0 = anchor, then the sampled drafts.
#   - Confidence head: sigmoid(linear(cat([post-norm hidden, markov emb(prev_j)])))
#     per row; the generator caps the draft window to the longest all-confident
#     prefix via params["draft_confidence_len"] (threshold EXL3_DSPARK_CONF).
#
# The backbone (input layer, blocks, dual-source KV, non-causal block attention,
# update_kv_from_target, generator flow) is inherited unchanged from DFlashModel;
# only the two heads and the block-output convention differ.


class DSparkConfidenceHead(Module):
    """
    AcceptRatePredictor: a single unquantized Linear(in_features -> 1) WITH bias.
    exl3's Linear has no bias support, so this is a small dedicated module.
    """

    def __init__(
        self,
        config,
        key: str,
        in_features: int,
    ):
        super().__init__(config, key, None)
        self.module_name = "DSparkConfidenceHead"
        self.in_features = in_features
        self.weight = None      # (1, in_features)
        self.bias = None        # (1,)
        self._numel = 0

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        self.weight = self.config.stc.get_tensor(
            self.key + ".weight", device, float2half = True
        )
        self.bias = self.config.stc.get_tensor(
            self.key + ".bias", device, float2half = True
        )
        self._numel = self.weight.numel() + self.bias.numel()

    @override
    def unload(self):
        self.device = None
        self.weight = None
        self.bias = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.weight": self.weight.contiguous(),
            f"{self.key}.bias": self.bias.contiguous(),
        }

    @override
    def weights_numel(self):
        return self._numel

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class DSparkDraftConfig(DFlashConfig):

    arch_string = "DSparkDraftModel"

    # SpecForge variant: dflash_config.target_layer_ids are used RAW (id i means
    # the output of target layer i). The original z-lab DFlash release needed +1.
    tap_shift = 0

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": DSparkDraftModel},
            **kwargs
        )

        # DSpark fields live at the top level and (duplicated) under dflash_config
        # in the published RadixArk config
        self.markov_rank = self.read_cfg(
            int, ["markov_rank", "dflash_config->markov_rank"], 0
        )
        self.markov_head_type = self.read_cfg(
            str, ["markov_head_type", "dflash_config->markov_head_type"], "vanilla"
        )
        self.enable_confidence_head = self.read_cfg(
            bool, ["enable_confidence_head", "dflash_config->enable_confidence_head"],
            False
        )
        self.confidence_head_with_markov = self.read_cfg(
            bool, ["confidence_head_with_markov", "dflash_config->confidence_head_with_markov"],
            True
        )

        assert self.markov_head_type == "vanilla", \
            f"DSparkDraftModel: only 'vanilla' markov_head_type is supported, " \
            f"got {self.markov_head_type!r}"
        if self.confidence_head_with_markov:
            assert self.markov_rank > 0, \
                "DSparkDraftModel: confidence_head_with_markov requires markov_rank > 0"
        assert self.enable_confidence_head == (self.markov_rank > 0 or
                                               self.enable_confidence_head), \
            "DSparkDraftModel: confidence head requires consistent config"


class DSparkDraftModel(DFlashModel):
    """
    DFlash backbone with DSpark Markov/confidence heads. All rows of the draft
    block produce draft tokens (next-token alignment), so sample_from_state
    returns [seed, d1..d_block] and default_draft_size = block_size, unlike the
    plain DFlash convention inherited from DFlashModel.
    """

    config_class = DSparkDraftConfig

    def __init__(
        self,
        config: DSparkDraftConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # Backbone modules end here; the heads below are not part of the forward pass
        self.fwd_end_idx = len(self.modules)

        # Markov bigram head (per-token logit bias in the sampling loop) and
        # confidence head (per-position accept-probability), unquantized
        self.markov_w1 = Embedding(
            config = config,
            key = "markov_head.markov_w1",
            vocab_size = config.vocab_size,
            hidden_size = config.markov_rank,
        )
        # Device-resident (~63 MB): the sampling loop stays on-stream with no host syncs
        self.markov_w1.caps["prefer_cpu"] = False
        self.markov_w2 = Linear(
            config = config,
            key = "markov_head.markov_w2",
            in_features = config.markov_rank,
            out_features = config.vocab_size,
            qmap = None,
        )
        conf_in = config.hidden_size
        if config.confidence_head_with_markov:
            conf_in += config.markov_rank
        self.confidence = DSparkConfidenceHead(
            config = config,
            key = "confidence_head.proj",
            in_features = conf_in,
        ) if config.enable_confidence_head else None
        self.modules += [m for m in (self.markov_w1, self.markov_w2, self.confidence) if m]

        # Every block row drafts (see class docstring); verify width = block_size + 1
        self.caps["default_draft_size"] = config.block_size

        # Draft length gate: keep the longest prefix with sigmoid(confidence) >= threshold
        self.draft_conf_threshold = float(os.environ.get("EXL3_DSPARK_CONF", "0.5"))
        self._conf_stats = [] if os.environ.get("EXL3_DSPARK_CONF_STATS") else None
        self._markov_w1_dev = None

    @override
    def forward(self, input_ids: torch.Tensor, params: dict, **kwargs) -> torch.Tensor:
        """Draft-block forward: seed token per row -> [seed, mask x (block-1)] ->
        input layer (embed via target + stream expand) -> blocks -> final norm.
        Returns the POST-norm state (bsz, block, hidden) fp16 for sample_from_state
        (which also feeds it to the confidence head)."""
        x = self.prepare_inputs(input_ids, params)
        params["dspark_seed_ids"] = input_ids
        for m in self.modules[:self.fwd_end_idx]:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
        return x

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        """Target lm_head over all block rows at once, then the sequential greedy
        loop with the markov bigram bias. Returns (bsz, block + 1) ids
        [seed, drafts...]; the generator crops the seed."""
        if not self.attached_model().loaded_tp:
            ll = self.attached_model().logit_layer_idx
            lm = self.attached_model().modules[ll]
            logits = lm.prepare_for_device(state, params)
            logits = lm.forward(logits, params)
            logits = logits[..., :self.attached_model().config.vocab_size]
        else:
            # TODO: TP target support (private embed/head copies as in
            # DeepseekV4MTPModel._load_own_embed_head)
            raise NotImplementedError(
                "DSparkDraftModel does not yet support tensor-parallel targets"
            )

        logits = logits.float()
        b, s, _ = logits.shape
        # Sequential in the sampled chain but fully on-device: embedding gather +
        # bias gemv + argmax per step, no host round trips
        dev = self.markov_w2.device
        logits = to_dev(logits, dev)
        seed = to_dev(params["dspark_seed_ids"], dev)
        if self._markov_w1_dev is None:
            self._markov_w1_dev = to_dev(self.markov_w1.embedding.weight.data, dev)
        w2 = self.markov_w2
        out = torch.empty((b, s + 1), dtype = torch.long, device = dev)
        out[:, 0] = seed[:, -1]
        embs = []
        for i in range(s):
            emb = F.embedding(out[:, i], self._markov_w1_dev).half()
            embs.append(emb)
            bias = w2.forward(emb.unsqueeze(1), params)
            logits[:, i] += bias[:, 0].float()
            out[:, i + 1] = torch.argmax(logits[:, i], dim = -1)

        # Confidence-capped draft length: proj(cat([post-norm hidden, markov emb]))
        # per position; the generator clamps its window to the longest all-confident
        # prefix (batch max), 0 = skip drafting this round
        if self.confidence is not None:
            cdev = self.confidence.device
            feats = [to_dev(state, cdev).half()]
            if self.config.confidence_head_with_markov:
                feats.append(to_dev(torch.stack(embs, dim = 1), cdev).half())
            conf = self.confidence.forward(torch.cat(feats, dim = -1), params)
            cs = torch.sigmoid(conf.float().squeeze(-1))
            if self._conf_stats is not None:
                self._conf_stats.append(cs[0].tolist())
            keep = cs >= self.draft_conf_threshold
            lens = torch.cumprod(keep.to(torch.int32), dim = 1).sum(dim = 1)
            params["draft_confidence_len"] = int(lens.max().item())

        return out
