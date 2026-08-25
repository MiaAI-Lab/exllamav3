#!/usr/bin/env python
"""
Generate a workload-matched calibration trace for EXL3 conversion.

Runs coding/math-heavy prompts (our serving mix) through the model — by default
the serving bundle (workload-matched 3.5bpw) *with* the DSpark drafter (lossless:
ssampled output is identical to the target alone, so the trace is a faithful
self-trace at ~2x speed) — sampling at 0.6 / top-k 20 / top-p 0.95 with thinking
enabled, and
captures prompt + response token IDs in turboderp's published cal_trace.json
format:

    {"model": ..., "vocab_size": ..., "template_vars": {...},
     "meta": {...},
     "rows": [{"input_ids": [...], "response_ids": [...], ...}, ...]}

The result is loadable by
exllamav3.conversion.calibration_data.load_calibration_trace() (i.e. by
convert.py -cd/--cal-data and tools/quantize_model.py --cal-data). Conventions
mirror the published trace: input_ids carry the full chat template ending in
"<|im_start|>assistant\\n<think>\\n", response_ids carry the sampled tokens with
the stop token stripped.

Prompt mix (PLAN.md §5.6 workloads, coding + math heavy):
    humaneval 164 + mbpp 164 + gsm8k 120 + math500 120 + alpaca 60 + mt-bench 40,
shuffled with seed 0, generated until --target-tokens is reached.

The run is resumable: completed conversations are appended line-by-line to
<out>.rows.jsonl; restarting the same command skips already-generated prompts
and continues. The final JSON is assembled (and validated against the loader)
when the token target is reached or the prompt pool is exhausted.

Usage:
  python tools/gen_cal_trace.py                       # ~620k tokens, bundle + draft
  python tools/gen_cal_trace.py --smoke               # 3 prompts x 96 tokens, /tmp out
  python tools/gen_cal_trace.py --no-draft            # generate without the drafter
  python tools/gen_cal_trace.py --reasoning-effort medium
  python tools/gen_cal_trace.py --finalize            # assemble final JSON from jsonl only

Env overrides (same as the other dspark tools): M2_TARGET_DIR, M2_DRAFT_DIR.
"""

import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dspark_m4_bench import WORKLOADS, make_sampler

TARGET_DIR = os.environ.get("M2_TARGET_DIR", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm")
DRAFT_DIR = os.environ.get("M2_DRAFT_DIR", "test_models/Qwen3.8-27B-exl3-3.5bpw-wm/draft")
DEFAULT_OUT = "test_models/sources/cal_trace_qwen38_27b_workload.json"

# (workload, prompts drawn) — coding + math reasoning heavy, small general slice
TRACE_POOL = [
    ("humaneval", 164),
    ("mbpp", 164),
    ("gsm8k", 120),
    ("math500", 120),
    ("alpaca", 60),
    ("mt-bench", 40),
]

# loader requirements for a full-size conversion (convert.py defaults)
CAL_ROWS, CAL_COLS = 250, 2048


def build_pool(spec):
    """Load the prompt pool: [(source, prompt), ...], shuffled with seed 0."""
    from datasets import load_dataset
    pool = []
    for name, count in spec:
        repo, kwargs = WORKLOADS[name]["load"]
        ds = load_dataset(repo, **kwargs)
        prompts = []
        for x in ds:
            prompts.append(WORKLOADS[name]["format"](x))
            if len(prompts) >= count:
                break
        pool += [(name, p) for p in prompts]
    random.seed(0)
    random.shuffle(pool)
    return pool


def build_generator(target_dir, draft_dir, use_draft, num_draft_tokens):
    from argparse import ArgumentParser
    from exllamav3 import model_init, Generator
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args=True)
    argv = ["-m", target_dir, "-gs", "110"]
    if use_draft:
        argv += ["-dm", draft_dir]
    args = parser.parse_args(argv)
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(args, progress=True)
    generator = Generator(
        model, cache, tokenizer,
        draft_model=draft_model if use_draft else None,
        draft_cache=draft_cache if use_draft else None,
        num_draft_tokens=num_draft_tokens if use_draft else None,
        record_draft_stats=True,
    )
    return generator, (draft_model if use_draft else None)


def run_conversation(generator, tokenizer, prompt, effort, max_new_tokens,
                     num_draft_tokens):
    """One sampled conversation; returns (input_ids, response_ids, stats)."""
    from exllamav3 import Job
    kw = dict(add_generation_prompt=True, enable_thinking=True)
    if effort:
        kw["reasoning_effort"] = effort
    input_ids = tokenizer.hf_chat_template(
        [{"role": "user", "content": prompt}], **kw)
    prompt_len = input_ids.shape[-1]
    job = Job(input_ids=input_ids, max_new_tokens=max_new_tokens,
              sampler=make_sampler(), seed=0)
    t0 = time.time()
    generator.enqueue(job)
    while generator.num_remaining_jobs():
        generator.iterate()
    dt = time.time() - t0
    seq = job.sequences[0]
    resp = seq.sequence_ids.torch_slice(prompt_len, None)[0].tolist()
    # draft jobs finish ~verify_width short of the cap (generator requeue
    # bookkeeping); anything at/above that is a cap stop, else a stop token
    if len(resp) >= max_new_tokens - num_draft_tokens - 1:
        eos_reason = "max_new_tokens"
    else:
        eos_reason = "stop_token"
    stats = {"seconds": round(dt, 2), "tok_per_s": round(len(resp) / dt, 2),
             "eos_reason": eos_reason}
    if job.draft_stats:
        acc = [s[2] + 1 for s in job.draft_stats]  # incl. bonus token
        stats["accept_length"] = round(sum(acc) / len(acc), 3)
    return input_ids[0].tolist(), resp, stats


def finalize(out_file, rows_file, model_field, effort, tokenizer_dir,
             cal_rows, cal_cols):
    """Assemble the final trace JSON from the jsonl rows and validate it."""
    rows = []
    with open(rows_file) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    in_tok = sum(len(r["input_ids"]) for r in rows)
    out_tok = sum(len(r["response_ids"]) for r in rows)
    # validate against the actual loader (also gives us the true vocab size)
    from exllamav3 import Config
    from exllamav3.conversion.calibration_data import load_calibration_trace
    from exllamav3.tokenizer import Tokenizer
    tok = Tokenizer(Config.from_directory(tokenizer_dir))
    trace = {
        "model": model_field,
        "vocab_size": tok.actual_vocab_size,
        "template_vars": {"enable_thinking": True,
                          **({"reasoning_effort": effort} if effort else {})},
        "meta": {
            "rows": len(rows),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "description": "workload-matched self-generated trace "
                           "(coding + math reasoning heavy, thinking on, "
                           "sampled 0.6/top-k 20/top-p 0.95, seed 0)",
            "sources": dict(sorted(_count_sources(rows).items())),
        },
        "rows": rows,
    }
    with open(out_file, "w") as f:
        json.dump(trace, f)
    print(f" == wrote {out_file}: {len(rows)} conversations, "
          f"{in_tok} input + {out_tok} output = {in_tok + out_tok} tokens")
    need = cal_rows * cal_cols
    if in_tok + out_tok < need:
        print(f" !! WARNING: {in_tok + out_tok} tokens < {need} "
              f"({cal_rows} x {cal_cols}) required by the converter; "
              f"continue the run before converting")
        return False
    t_rows = load_calibration_trace(out_file, cal_rows, cal_cols, tok)
    print(f" == loader validation PASS: {len(t_rows)} rows x {cal_cols} columns")
    return True


def _count_sources(rows):
    from collections import Counter
    return Counter(r.get("source", "?") for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target-tokens", type=int, default=620000,
                    help="stop once this many input+response tokens are captured")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["xhigh", "medium", "low"],
                    help="chat-template reasoning effort (default: template "
                         "default = xhigh, our serving convention; 'medium' "
                         "matches turboderp's published trace)")
    ap.add_argument("--no-draft", action="store_true",
                    help="generate without the DSpark drafter (slower)")
    ap.add_argument("--num-draft-tokens", type=int, default=7)
    ap.add_argument("--smoke", action="store_true",
                    help="3 prompts x 96 tokens to /tmp, quick end-to-end check")
    ap.add_argument("--finalize", action="store_true",
                    help="only assemble + validate the final JSON from existing "
                         "jsonl rows (no generation)")
    ap.add_argument("--cal-rows", type=int, default=CAL_ROWS)
    ap.add_argument("--cal-cols", type=int, default=CAL_COLS)
    args = ap.parse_args()

    if args.smoke:
        args.out = "/tmp/cal_trace_smoke.json"
        args.target_tokens = 10 ** 9     # cap by prompt count instead
        args.max_new_tokens = 96
        args.cal_rows, args.cal_cols = 1, 96

    rows_file = args.out + ".rows.jsonl"
    use_draft = not args.no_draft
    model_field = TARGET_DIR + (" + DSpark draft" if use_draft else " (no draft)")

    if args.finalize:
        ok = finalize(args.out, rows_file, model_field, args.reasoning_effort,
                      TARGET_DIR, args.cal_rows, args.cal_cols)
        sys.exit(0 if ok else 1)

    import torch
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f" == building prompt pool (coding + math heavy) ...", flush=True)
    pool = build_pool(TRACE_POOL)
    if args.smoke:
        pool = pool[:3]
    print(f"    {len(pool)} prompts: "
          + ", ".join(f"{n}={c}" for n, c in TRACE_POOL), flush=True)

    # resume: skip prompts already completed in the jsonl
    done = 0
    in_tok = out_tok = 0
    if os.path.exists(rows_file):
        with open(rows_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done += 1
                    in_tok += len(r["input_ids"])
                    out_tok += len(r["response_ids"])
        print(f" == resuming: {done} conversations, {in_tok + out_tok} tokens "
              f"already in {rows_file}", flush=True)

    generator, draft_model = build_generator(TARGET_DIR, DRAFT_DIR, use_draft,
                                             args.num_draft_tokens)
    tokenizer = generator.tokenizer

    t_start = time.time()
    gen_tokens = 0
    f = open(rows_file, "a")
    try:
        for idx in range(done, len(pool)):
            source, prompt = pool[idx]
            ids, resp, stats = run_conversation(
                generator, tokenizer, prompt, args.reasoning_effort,
                args.max_new_tokens, args.num_draft_tokens)
            row = {"conversation": idx, "turn": 0, "epoch": 0,
                   "source": source, "eos_reason": stats["eos_reason"],
                   "input_ids": ids, "response_ids": resp}
            f.write(json.dumps(row) + "\n")
            f.flush()
            in_tok += len(ids)
            out_tok += len(resp)
            gen_tokens += len(ids) + len(resp)
            total = in_tok + out_tok
            rate = gen_tokens / max(time.time() - t_start, 1e-9)
            eta_s = (args.target_tokens - total) / rate if rate > 0 else 0
            eta = f"  ~{eta_s / 60:.0f} min to target" if args.target_tokens < 10 ** 8 else ""
            acc = f"  accept={stats['accept_length']}" if "accept_length" in stats else ""
            print(f" [{idx + 1}/{len(pool)}] {source:9s} in={len(ids):4d} out={len(resp):4d} "
                  f"({stats['tok_per_s']} tok/s{acc})  total={total}{eta}", flush=True)
            if total >= args.target_tokens:
                print(f" == token target {args.target_tokens} reached", flush=True)
                break
    except KeyboardInterrupt:
        print(f" == interrupted; {in_tok + out_tok} tokens saved to {rows_file}; "
              f"re-run the same command to resume", flush=True)
        sys.exit(1)
    finally:
        f.close()

    ok = finalize(args.out, rows_file, model_field, args.reasoning_effort,
                  TARGET_DIR, args.cal_rows, args.cal_cols)
    if not ok and not args.smoke:
        print(" (run more prompts or lower --cal-rows/--cal-cols)", flush=True)


if __name__ == "__main__":
    main()
