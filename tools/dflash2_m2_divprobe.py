#!/usr/bin/env python3
"""M2 divergence probe: on one prompt, greedy no-draft vs greedy with-draft.
Captures per-step top-5 logits on the no-draft run (via job.logits) and
reports the top-2 gap at the first divergence."""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler.presets import ArgmaxSampler

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n -",
    "Write a short paragraph explaining why the sky is blue.",
    "The three laws of robotics were first stated by",
    "import numpy as np\n\ndef moving_average(x, w):\n    return",
]

def main():
    pi = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 92
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = True)
    args = parser.parse_args([
        "-m", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm",
        "-gs", "110",
        "-dm", "test_models/sources/Qwen3.8-27B-DFlash2",
    ])
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(args, progress = True)

    def run(draft):
        generator = Generator(model, cache, tokenizer,
                              draft_model = draft_model if draft else None,
                              draft_cache = draft_cache if draft else None,
                              num_draft_tokens = 7, record_draft_stats = True)
        input_ids = tokenizer.encode(PROMPTS[pi])
        prompt_len = input_ids.shape[-1]
        job = Job(input_ids = input_ids, max_new_tokens = n_tok,
                  sampler = ArgmaxSampler(), identifier = pi,
                  return_logits = not draft)
        generator.enqueue(job)
        tops = []
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("stage") == "stream":
                    lg = r.get("logits")
                    if lg is not None:
                        lg = lg[0].float()
                        if lg.dim() == 2:
                            for row in lg:
                                tops.append(torch.topk(row, 5).values.cpu())
                        else:
                            tops.append(torch.topk(lg, 5).values.cpu())
        seq = job.sequences[0]
        toks = seq.sequence_ids.torch_slice(prompt_len, None).tolist()
        return toks, tops

    plain, tops = run(False)
    spec, _ = run(True)
    n = min(len(plain), len(spec))
    first = next((i for i in range(n) if plain[i] != spec[i]), None)
    print(f"p{pi}: plain {len(plain)} tok, spec {len(spec)} tok, first divergence at token {first}")
    if first is not None and first < len(tops):
        t = tops[first]
        print(f"top-5 logits: {[round(v, 3) for v in t.tolist()]}")
        print(f"top-2 gap = {float(t[0] - t[1]):.4f}  (near-tie if small)")
        print(f"plain token {plain[first]} vs spec token {spec[first]}")

if __name__ == "__main__":
    main()
