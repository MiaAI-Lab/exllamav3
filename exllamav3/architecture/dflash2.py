from __future__ import annotations
from typing_extensions import override
import torch

from .dflash import DFlashConfig, DFlashModel
from ..model.config import no_default
from ..model.model import Model
from ..modules import RMSNorm, Attention, GatedMLP
from ..modules.arch_specific.dflash2 import (
    DFlash2Block, DFlash2DynConv, DFlash2Selector, _DFlash2Norm,
)

# DFlash2 draft model (z-lab/Qwen3.8-27B-DFlash2): DFlash1 backbone + grouped
# dynamic convolutions around every attention/MLP + a top-16 candidate
# selector replacing per-row argmax. Reference: ``dflash`` pip package
# (dflash.model.DFlash2DraftModel); semantics pinned by
# tools/dflash2_cache_trace.py (cache/positions/mask) and
# tools/dflash2_unit_tests.py (conv causality, selector, rejection sampling).
#
# Conventions:
#   - Block input = [anchor, mask x (block_size-1)]; rows 1..7 predict their
#     OWN position (z-lab diffusion convention, unlike DSpark next-token
#     alignment); the selector walks top-16 candidates from the anchor.
#   - Taps: reference reads HF hidden_states[target_layer_ids[i] + 1] = output
#     of target layer id  =>  tap_shift 0 (exl3 export index = layer output).
#   - Draft cache ctx K/V come from update_kv_from_target (fc+hidden_norm
#     projected tap stream), inherited unchanged from the DFlash1 backbone;
#     per-round writes at the new cache position overwrite the transient
#     noise-block K/V, reproducing the reference's crop semantics.
#   - Proposals are the greedy selector path (T-independent), verified by the
#     stock accept-while-match rule — trivially lossless at every temperature
#     (per-position output marginal equals the target distribution).


class DFlash2Config(DFlashConfig):

    arch_string = "DFlash2DraftModel"

    # Reference extract uses hidden_states[id + 1] == output of layer id
    tap_shift = 0

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": DFlash2Model},
            **kwargs
        )

        self.conv_kernel_size = self.read_cfg(
            int, ["dflash_config->conv_kernel_size", "conv_kernel_size"], 2)
        self.conv_group_size = self.read_cfg(
            int, ["dflash_config->conv_group_size", "conv_group_size"], 16)
        self.selector_rank = self.read_cfg(
            int, ["dflash_config->selector_rank", "selector_rank"], no_default)
        self.selector_top_k = self.read_cfg(
            int, ["dflash_config->selector_top_k", "selector_top_k"], no_default)


class DFlash2Model(DFlashModel):
    """DFlash1 backbone rebuilt with dynconv-wrapped blocks + selector head."""

    config_class = DFlash2Config

    def __init__(
        self,
        config: DFlash2Config,
        **kwargs
    ):
        # Build the DFlash1 backbone (input layer, plain blocks, final norm,
        # caps, attach/update_kv_from_target machinery)
        super().__init__(config, **kwargs)

        # Swap each TransformerBlock for a conv-wrapped DFlash2Block reusing
        # the same attn/mlp modules (attn_modules already reference them, so
        # update_kv_from_target keeps working unchanged). Norms are replaced
        # with torch-only bf16-capable norms (residual stream is bf16).
        for idx in range(config.num_hidden_layers):
            old = self.modules[self.first_block_idx + idx]
            self.modules[self.first_block_idx + idx] = DFlash2Block(
                config = config,
                key = f"layers.{idx}",
                layer_idx = idx,
                attn = old.attn,
                mlp = old.mlp,
                attn_norm = _DFlash2Norm(
                    config = config,
                    key = f"layers.{idx}.input_layernorm",
                    eps = config.rms_norm_eps,
                ),
                mlp_norm = _DFlash2Norm(
                    config = config,
                    key = f"layers.{idx}.post_attention_layernorm",
                    eps = config.rms_norm_eps,
                ),
                attn_conv = DFlash2DynConv(
                    config = config,
                    key = f"layers.{idx}.attention_conv",
                    hidden_size = config.hidden_size,
                    kernel_size = config.conv_kernel_size,
                    group_size = config.conv_group_size,
                ),
                mlp_conv = DFlash2DynConv(
                    config = config,
                    key = f"layers.{idx}.mlp_conv",
                    hidden_size = config.hidden_size,
                    kernel_size = config.conv_kernel_size,
                    group_size = config.conv_group_size,
                ),
            )

        # Backbone (fwd path) ends at the final norm; the selector is a head
        self.fwd_end_idx = len(self.modules)
        self.modules[-1] = _DFlash2Norm(
            config = config,
            key = "norm",
            eps = config.rms_norm_eps,
        )

        self.selector = DFlash2Selector(
            config = config,
            key = "candidate_selector",
            vocab_size = config.vocab_size,
            hidden_size = config.hidden_size,
            rank = config.selector_rank,
            top_k = config.selector_top_k,
        )
        self.modules += [self.selector]

    @override
    def forward(self, input_ids: torch.Tensor, params: dict, **kwargs) -> torch.Tensor:
        """Draft-block forward: [anchor] -> input layer -> conv blocks ->
        final norm. Returns the post-norm state (bsz, block, hidden) for
        sample_from_state."""
        x = self.prepare_inputs(input_ids, params)
        params["dflash2_anchor_ids"] = input_ids
        # Bilateral draft attention (reference is_causal=False + sliding_window):
        # the whole noise block attends to itself bidirectionally plus a left
        # window over the cached context. Expressed as one non-causal span;
        # get_window_size()'s right=0 alone (z-lab DFlash1 semantics) would
        # wrongly mask future noise rows.
        params["non_causal_spans"] = [(0, self.config.block_size, True)]
        for m in self.modules[:self.fwd_end_idx]:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
        return x

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        """Target lm_head over all block rows, then the selector walk over
        rows 1.. (rows predict their own position; row 0 is the anchor).
        Returns (bsz, block) ids [anchor, path...]; the generator crops the
        anchor. Greedy walk at every temperature (lossless: verification is
        accept-while-match against target samples)."""
        if self.attached_model().loaded_tp:
            raise NotImplementedError(
                "DFlash2Model does not yet support tensor-parallel targets"
            )
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        logits = lm.prepare_for_device(state.half(), params)
        logits = lm.forward(logits, params)
        logits = logits[..., :self.attached_model().config.vocab_size]

        dev = self.selector.device
        anchor = params["dflash2_anchor_ids"][:, -1].to(dev)
        temperature = params.get("draft_temperature", 0.0)
        rand_u32 = params.get("draft_rand_u32")
        if temperature and temperature > 0.0:
            path, q, q_full, cands = self.selector.walk(
                state[:, 1:].to(dev), logits[:, 1:].to(dev).float(), anchor,
                temperature = temperature, rand_u32 = rand_u32 if rand_u32 is not None else 0,
            )
            params["draft_q"] = q
            params["draft_q_full"] = q_full
            params["draft_cands"] = cands
        else:
            path = self.selector.walk(state[:, 1:].to(dev), logits[:, 1:].to(dev).float(), anchor)
            params["draft_q"] = None
        out = torch.empty(
            (path.shape[0], path.shape[1] + 1),
            dtype = torch.long, device = dev)
        out[:, 0] = anchor
        out[:, 1:] = path
        return out

    @classmethod
    @override
    def get_additional_compiled_tensors(cls, config: DFlash2Config) -> dict:
        # Backbone fc norm (DFlash1) + conv base kernels + selector codebooks
        tensors = dict(config.stc.list_tensors(prefix = cls.key_fc_norm))
        tensors.update(config.stc.list_tensors(prefix = "candidate_selector."))
        for idx in range(config.num_hidden_layers):
            for conv in ("attention_conv", "mlp_conv"):
                tensors.update(config.stc.list_tensors(
                    prefix = f"layers.{idx}.{conv}.base_kernel"))
        return tensors
