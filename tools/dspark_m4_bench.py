#!/usr/bin/env python
"""
M4 benchmark harness: per-workload spec acceptance + decode speed for the
DSpark speculative-decoding bundle, following the published RadixArk/dflash
measurement convention:

  sampling temperature 0.6, top-k 20, top-p 0.95; thinking enabled;
  seed 0; spec_accept_length includes the target bonus token.

Usage (defaults point at the serving bundle — workload-matched 3.5bpw):
  python tools/dspark_m4_bench.py                        # draft runs, all workloads
  python tools/dspark_m4_bench.py --baseline             # no-draft baseline runs
  python tools/dspark_m4_bench.py --sweep-conf           # EXL3_DSPARK_CONF sweep
  python tools/dspark_m4_bench.py --workloads gsm8k,mt-bench --num-prompts 8

Results are appended to test_models/dspark-evals/ and printed as a summary
table with the published reference numbers.
"""

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_DIR = os.environ.get("M2_TARGET_DIR", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm")
DRAFT_DIR = os.environ.get("M2_DRAFT_DIR", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm/draft")
EVALS_DIR = "test_models/dspark-evals"

# (dataset args, format, published accept length)
WORKLOADS = {
    "humaneval": {
        "load": ("openai/openai_humaneval", {"split": "test"}),
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n" + x["prompt"] + "\n```",
        "published": 3.47,
    },
    "gsm8k": {
        "load": ("openai/gsm8k", {"split": "test", "name": "main"}),
        "format": lambda x: x["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "published": 4.57,
    },
    "math500": {
        "load": ("HuggingFaceH4/MATH-500", {"split": "test"}),
        "format": lambda x: x["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "published": 4.08,
    },
    "mbpp": {
        "load": ("google-research-datasets/mbpp", {"split": "test", "name": "sanitized"}),
        "format": lambda x: x["prompt"],
        "published": 3.67,
    },
    "mt-bench": {
        "load": ("HuggingFaceH4/mt_bench_prompts", {"split": "train"}),
        "format": lambda x: x["prompt"][0],   # first turn only (see report note)
        "published": 3.10,
    },
    "alpaca": {
        "load": ("yahma/alpaca-cleaned", {"split": "train"}),
        "format": lambda x: x["instruction"] + ("\n\n" + x["input"] if x.get("input") else ""),
        "published": 2.95,
    },
}


def load_prompts(name, num_prompts):
    from datasets import load_dataset
    cfg = WORKLOADS[name]
    repo, kwargs = cfg["load"]
    ds = load_dataset(repo, **kwargs)
    prompts = []
    for x in ds:
        prompts.append(cfg["format"](x))
        if len(prompts) >= num_prompts:
            break
    return prompts


def build_generator(draft: bool, num_draft_tokens: int = 7):
    from argparse import ArgumentParser
    from exllamav3 import model_init, Generator
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = True)
    argv = ["-m", TARGET_DIR, "-gs", "110"] + os.environ.get("M4_EXTRA_ARGS", "").split()
    if draft:
        argv += ["-dm", DRAFT_DIR]
    args = parser.parse_args(argv)
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(args, progress = True)
    generator = Generator(
        model, cache, tokenizer,
        draft_model = draft_model if draft else None,
        draft_cache = draft_cache if draft else None,
        num_draft_tokens = num_draft_tokens if draft else None,
        record_draft_stats = True,
    )
    return generator, (draft_model if draft else None)


def make_sampler(temperature = None, top_k = None, top_p = None):
    from exllamav3.generator.sampler.presets import ComboSampler
    return ComboSampler(temperature = make_sampler.t or 0.6,
                        top_k = make_sampler.k or 20,
                        top_p = make_sampler.p or 0.95)
make_sampler.t = None
make_sampler.k = None
make_sampler.p = None


def run_workload(generator, tokenizer, name, prompts, max_new_tokens):
    """Sequential per-request runs; per-request accept length, decode tok/s, TTFT."""
    from exllamav3 import Job
    per_request = []
    for p in prompts:
        input_ids = tokenizer.hf_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt = True,
            enable_thinking = True,
        )
        job = Job(input_ids = input_ids, max_new_tokens = max_new_tokens,
                  sampler = make_sampler(), seed = 0)
        t0 = time.time()
        ttft = None
        generator.enqueue(job)
        seen = 0
        while generator.num_remaining_jobs():
            results = generator.iterate()
            if ttft is None and any(r.get("text") for r in results):
                ttft = time.time() - t0
        dt = time.time() - t0
        seq = job.sequences[0]
        n_new = seq.sequence_ids.seq_len - input_ids.shape[-1]
        rec = {"tokens": int(n_new), "seconds": round(dt, 2),
               "tok_per_s": round(n_new / dt, 2), "ttft_s": round(ttft, 2) if ttft else None}
        if job.draft_stats:
            acc = [s[2] + 1 for s in job.draft_stats]      # incl. bonus token
            rec["verify_rounds"] = len(acc)
            rec["accept_length"] = round(sum(acc) / len(acc), 3)
        per_request.append(rec)
    accs = [r["accept_length"] for r in per_request if "accept_length" in r]
    tps = [r["tok_per_s"] for r in per_request]
    ttfts = [r["ttft_s"] for r in per_request if r["ttft_s"]]
    summary = {
        "workload": name,
        "num_prompts": len(per_request),
        "max_new_tokens": max_new_tokens,
        "mean_accept_length": round(sum(accs) / len(accs), 3) if accs else None,
        "mean_tok_per_s": round(sum(tps) / len(tps), 2),
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "total_tokens": sum(r["tokens"] for r in per_request),
    }
    return summary, per_request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default = ",".join(WORKLOADS))
    ap.add_argument("--num-prompts", type = int, default = 24)
    ap.add_argument("--max-new-tokens", type = int, default = 512)
    ap.add_argument("--baseline", action = "store_true", help = "run no-draft baseline")
    ap.add_argument("--temperature", type = float, default = 0.6)
    ap.add_argument("--top-k", type = int, default = 20)
    ap.add_argument("--top-p", type = float, default = 0.95)
    ap.add_argument("--tag", default = "")
    ap.add_argument("--sweep-conf", action = "store_true",
                    help = "EXL3_DSPARK_CONF threshold sweep on the first workload")
    args = ap.parse_args()
    make_sampler.t = args.temperature
    make_sampler.k = args.top_k
    make_sampler.p = args.top_p

    import torch
    torch.manual_seed(0)
    os.makedirs(EVALS_DIR, exist_ok = True)

    mode = "baseline" if args.baseline else "draft"
    names = [w.strip() for w in args.workloads.split(",") if w.strip()]

    generator, draft_model = build_generator(draft = not args.baseline)
    from exllamav3 import Tokenizer  # noqa (tokenizer already loaded)
    tokenizer = generator.tokenizer

    run = {"mode": mode, "target": TARGET_DIR, "draft": DRAFT_DIR if not args.baseline else None,
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "sampling": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "seed": 0,
                        "thinking": True, "max_new_tokens": args.max_new_tokens},
           "results": []}

    if args.sweep_conf:
        name = names[0]
        prompts = load_prompts(name, max(8, args.num_prompts // 2))
        print(f" == confidence sweep on {name} ({len(prompts)} prompts)")
        sweep = []
        for conf in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            draft_model.draft_conf_threshold = conf
            summary, _ = run_workload(generator, tokenizer, f"{name}@conf={conf}", prompts,
                                      args.max_new_tokens)
            summary["conf"] = conf
            sweep.append(summary)
            print(f"   conf={conf}: accept={summary['mean_accept_length']}  "
                  f"{summary['mean_tok_per_s']} tok/s  ttft={summary['mean_ttft_s']}s")
        run["results"].append({"sweep": sweep})
    else:
        for name in names:
            prompts = load_prompts(name, args.num_prompts)
            print(f" == {mode} {name}: {len(prompts)} prompts x {args.max_new_tokens} max tokens")
            summary, per_request = run_workload(generator, tokenizer, name, prompts,
                                                args.max_new_tokens)
            summary["per_request"] = per_request
            if not args.baseline:
                summary["published_accept"] = WORKLOADS[name]["published"]
                if summary["mean_accept_length"] is not None:
                    rel = summary["mean_accept_length"] / summary["published_accept"] - 1
                    summary["rel_vs_published"] = f"{rel:+.1%}"
            run["results"].append(summary)
            pub = f" (published {WORKLOADS[name]['published']})" if not args.baseline else ""
            print(f"   accept={summary['mean_accept_length']}{pub}  "
                  f"{summary['mean_tok_per_s']} tok/s  ttft={summary['mean_ttft_s']}s  "
                  f"tokens={summary['total_tokens']}")

    tag = f"_{args.tag}" if args.tag else ""
    out_file = os.path.join(EVALS_DIR, f"dflash2_m4_{mode}{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(run, f, indent = 2)
    print(f" == wrote {out_file}")


if __name__ == "__main__":
    main()
