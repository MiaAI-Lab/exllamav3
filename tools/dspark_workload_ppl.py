#!/usr/bin/env python
"""
Workload-matched perplexity for EXL3 bundle comparison.

Same methodology as eval/ppl.py (fixed 2048-token rows, exact log-probs via
compute_target_log_probs) but the eval text is our serving mix instead of
wiki2: canonical solutions from HumanEval + GSM8K + MATH-500. These texts are
disjoint from every calibration corpus by construction (the default corpus is
generic text; the workload trace contains *model-generated* responses, never
the canonical solutions), so the comparison between bundles is fair.

Usage:
  python tools/dspark_workload_ppl.py -m <model_dir> [-gs 110] [--rows 64]
"""

import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset


def build_eval_text():
    parts = []
    # HumanEval: prompt + canonical solution (all 164)
    ds = load_dataset("openai/openai_humaneval", split="test")
    for x in ds:
        parts.append(x["prompt"] + x["canonical_solution"])
    # GSM8K: question + solution (first 400 of test)
    ds = load_dataset("openai/gsm8k", split="test", name="main")
    for x in list(ds)[:400]:
        parts.append(x["question"] + "\n" + x["answer"])
    # MATH-500: problem + solution (first 300)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for x in list(ds)[:300]:
        sol = x["solution"] if isinstance(x.get("solution"), str) else ""
        parts.append(x["problem"] + "\n" + sol)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    from exllamav3 import model_init
    model_init.add_args(ap, cache=False)
    ap.add_argument("--rows", type=int, default=64, help="number of 2048-token rows")
    args = ap.parse_args()

    import torch
    from exllamav3.util.measures import compute_target_log_probs

    print(f" == building workload eval text (humaneval + gsm8k + math500) ...", flush=True)
    text = build_eval_text()
    print(f"    {len(text)} chars", flush=True)

    model, config, _, tokenizer = model_init.init(
        args, override_dynamic_seq_len=2048, max_output_size=2048, max_output_factor=5)

    tokens = tokenizer.encode(text)[0].tolist()
    eval_len = 2048
    rows = []
    for a in range(0, len(tokens) - eval_len, eval_len):
        rows.append(tokens[a:a + eval_len])
        if len(rows) >= args.rows:
            break
    print(f"    {len(rows)} rows x {eval_len} tokens "
          f"({len(rows) * eval_len} tokens evaluated)", flush=True)

    vocab_size = tokenizer.actual_vocab_size
    logprob_sum, logprob_count = 0.0, 0
    for i, row in enumerate(rows):
        input_ids = torch.tensor([row])
        logits = model.forward(input_ids, {"attn_mode": "flash_attn_nc"})
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:].to(logits.device)
        tlp = compute_target_log_probs(logits, target_ids, vocab_size)
        logprob_sum += tlp.sum().item()
        logprob_count += target_ids.numel()
        del logits, tlp, target_ids, input_ids
        torch.cuda.empty_cache()
        print(f" [{i + 1}/{len(rows)}] running PPL = "
              f"{math.exp(-logprob_sum / logprob_count):.4f}", flush=True)

    ppl = math.exp(-logprob_sum / logprob_count)
    print(f" == WORKLOAD PPL ({args.model_dir}): {ppl:.6f} "
          f"over {logprob_count} tokens", flush=True)


if __name__ == "__main__":
    main()
