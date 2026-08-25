#!/usr/bin/env python
"""
M2 integration gate: speculative decoding with the DSpark draft model is LOSSLESS.

Runs greedy generation on a fixed prompt set twice (once with the draft model,
once without — controlled by --draft) and writes the generated token IDs plus
draft statistics to a JSON file. Compare the two runs with --compare.

Usage:
  python tools/dspark_m2_integration.py --no-draft --out /tmp/m2_nodraft.json
  python tools/dspark_m2_integration.py --draft    --out /tmp/m2_draft.json
  python tools/dspark_m2_integration.py --compare /tmp/m2_nodraft.json /tmp/m2_draft.json
"""

import argparse
import json
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_DIR = "test_models/sources/Qwen3.8-27B"
DRAFT_DIR = "test_models/sources/Qwen3.8-27B-DSpark"

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n -",
    "Write a short paragraph explaining why the sky is blue.",
    "The three laws of robotics were first stated by",
    "import numpy as np\n\ndef moving_average(x, w):\n    return",
    "The mitochondria is the",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "In 1492, Christopher Columbus",
    "The time complexity of binary search is",
    "Dear hiring manager, I am writing to apply for",
    "The Pythagorean theorem states that",
    "A step-by-step recipe for chocolate chip cookies:",
    "The main difference between TCP and UDP is",
    "function debounce(fn, delay) {",
    "Water boils at standard atmospheric pressure at a temperature of",
    "The first element of the periodic table is",
    "git rebase and git merge differ in that",
    "An FFT computes",
    "The largest planet in the solar system is",
    "To reverse a linked list iteratively, you",
]

DEFAULT_TARGET_DIR = "test_models/sources/Qwen3.8-27B"
DEFAULT_DRAFT_DIR = "test_models/sources/Qwen3.8-27B-DSpark"


def run(mode: str, out_file: str, max_new_tokens: int, num_draft_tokens: int):
    from argparse import ArgumentParser
    from exllamav3 import model_init, Generator, Job
    from exllamav3.generator.sampler.presets import ArgmaxSampler

    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = True)
    argv = ["-m", os.environ.get("M2_TARGET_DIR", DEFAULT_TARGET_DIR), "-gs", "110"]
    if mode == "draft":
        argv += ["-dm", os.environ.get("M2_DRAFT_DIR", DEFAULT_DRAFT_DIR)]
    args = parser.parse_args(argv)

    print(f" == loading ({mode}) ...")
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(args, progress = True)

    generator = Generator(
        model, cache, tokenizer,
        draft_model = draft_model if mode == "draft" else None,
        draft_cache = draft_cache if mode == "draft" else None,
        num_draft_tokens = num_draft_tokens,
        record_draft_stats = True,
    )

    results = []
    for i, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt)
        prompt_len = input_ids.shape[-1]
        job = Job(
            input_ids = input_ids,
            max_new_tokens = max_new_tokens,
            sampler = ArgmaxSampler(),
            identifier = i,
        )
        t0 = time.time()
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            generator.iterate()
        dt = time.time() - t0
        seq = job.sequences[0]
        new_ids = seq.sequence_ids.torch_slice(prompt_len, None)
        n = new_ids.shape[-1]
        stats = {
            "tokens": n,
            "seconds": round(dt, 2),
            "tok_per_s": round(n / dt, 2),
        }
        if job.draft_stats:
            accepted = [s[2] + 1 for s in job.draft_stats]   # incl. bonus token
            stats["verify_rounds"] = len(job.draft_stats)
            stats["mean_accept_length"] = round(sum(accepted) / len(accepted), 3)
        text = tokenizer.decode(new_ids, decode_special_tokens = True)
        if isinstance(text, list): text = text[0]
        results.append({
            "prompt": prompt,
            "ids": new_ids[0].tolist(),
            "text": text,
            "stats": stats,
        })
        print(f" [{i + 1}/{len(PROMPTS)}] {n} tokens in {dt:.1f}s "
              f"({stats['tok_per_s']} tok/s){' accept=' + str(stats.get('mean_accept_length')) if 'mean_accept_length' in stats else ''}")
        print(f"     {text[:100].replace(chr(10), ' ')}...")

    with open(out_file, "w") as f:
        json.dump({"mode": mode, "results": results}, f)
    print(f" == wrote {out_file}")


def compare(file_a: str, file_b: str):
    with open(file_a) as f: a = json.load(f)
    with open(file_b) as f: b = json.load(f)
    assert len(a["results"]) == len(b["results"])
    all_match = True
    for i, (ra, rb) in enumerate(zip(a["results"], b["results"])):
        ia, ib = ra["ids"], rb["ids"]
        n = min(len(ia), len(ib))
        # Draft jobs stop ~num_draft_tokens short of max_new_tokens (generator requeues
        # conservatively for any draft model); the gate is exact identity over the
        # common prefix
        match = ia[:n] == ib[:n]
        all_match &= match
        sa, sb = ra["stats"], rb["stats"]
        print(f" [{i}] tokens {sa['tokens']}/{sb['tokens']}  "
              f"{sa['tok_per_s']} vs {sb['tok_per_s']} tok/s  "
              f"accept={sb.get('mean_accept_length', '-')}  "
              f"{'MATCH (prefix identical)' if match else 'MISMATCH'}")
        if not match:
            for j in range(n):
                if ia[j] != ib[j]:
                    print(f"     first divergence at token {j}: {ia[j]} vs {ib[j]}")
                    break
    print()
    accepts = [r["stats"].get("mean_accept_length") for r in b["results"]]
    accepts = [x for x in accepts if x is not None]
    if accepts:
        print(f" mean accept length: {sum(accepts) / len(accepts):.3f} "
              f"(published SGLang mean: 3.35-3.40)")
    print("M2 LOSSLESS GATE:", "PASS" if all_match else "FAIL")
    return 0 if all_match else 1


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--draft", action = "store_true")
    mode.add_argument("--no-draft", dest = "no_draft", action = "store_true")
    parser.add_argument("--out", type = str, default = None)
    parser.add_argument("--compare", nargs = 2, metavar = ("NODRAFT", "DRAFT"))
    parser.add_argument("--max-new-tokens", type = int, default = 100)
    parser.add_argument("--num-draft-tokens", type = int, default = 7)
    args = parser.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))
    if args.draft:
        assert args.out, "--out required with --draft"
        run("draft", args.out, args.max_new_tokens, args.num_draft_tokens)
    elif args.no_draft:
        assert args.out, "--out required with --no-draft"
        run("nodraft", args.out, args.max_new_tokens, 0)
    else:
        parser.error("specify --draft, --no-draft or --compare")


if __name__ == "__main__":
    main()
