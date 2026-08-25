"""
CLI capability manifest for the EXL3 web console.

Every entry describes one CLI surface of this repo (script + arguments) in a
form the frontend can render as a form and the server can translate into an
argv. This keeps the UI a faithful 1:1 wrapper of the CLI: same flags, same
defaults, same scripts.

Field kinds:
  str    free text            int / float   number
  bool   checkbox (store_true flag)
  choice select (choices list)
  path   text with model-dir suggestions on the client
  env    environment variable (not a CLI flag)

Field flags:
  "positional": True  -> value appended without a flag (qbench project yaml,
                         longctx_passkey ctx/depths/kv-bits)
"""
import platform
import re

# Environment pinned on GB10/aarch64 (auto-detect fails there), mirroring
# quantize.sh; x86 auto-detects and needs no pinning.
QUANT_ENV = ({"TORCH_CUDA_ARCH_LIST": "12.0;12.1",
              "CUDA_HOME": "/usr/local/cuda"}
             if platform.machine() == "aarch64" else {})


def _flag_from_label(label, name, positional):
    """Derive the CLI flag from the label: 'Cache quant (-cq)' -> '-cq',
    '--override_anyway' -> itself. Env fields use their own name."""
    m = re.search(r"\((--?[\w][\w-]*)\)", label)
    if m:
        return m.group(1)
    if label.strip().startswith("-"):
        return label.strip()
    return name


def f(name, label, kind="str", default=None, choices=None, required=False,
      help="", flag=None, placeholder=None, positional=False):
    if flag is None:
        flag = name if kind == "env" else _flag_from_label(label, name, positional)
    return {"name": name, "label": label, "kind": kind, "default": default,
            "choices": choices, "required": required, "help": help,
            "flag": flag, "placeholder": placeholder,
            "positional": positional}


# --- model_init shared subset (base/cache/draft), used by eval scripts -----

def mi_fields(draft=False, cache_default=None):
    out = [
        f("model_dir", "Model dir (-m)", "path", required=True,
          help="Path to the model directory"),
        f("gpu_split", "VRAM cap GB (-gs)", "str",
          help="Maximum VRAM to use per device, in GB"),
    ]
    if cache_default is not None:
        out += [
            f("cache_size", "Cache tokens (-cs)", "int", default=cache_default,
              help="Total KV cache size in tokens"),
            f("cache_quant", "Cache quant (-cq)", "str",
              help="kv_bits or k_bits,v_bits (e.g. 8 / 8,4 / fp8 / nvfp4)"),
            f("cpu_cache_size", "CPU cache GB (-ccs)", "float",
              help="Second-tier CPU cache in GB"),
        ]
    if draft:
        out += [
            f("draft_model_dir", "Draft dir (-dm)", "path",
              help="Draft model directory for speculative decoding"),
            f("num_draft_tokens", "Draft tokens (-ndt)", "int",
              help="Number of draft tokens (draft model default, else 4)"),
        ]
    return out


QUANT_FIELDS = [
    f("model", "Source model", "path", required=True,
      help="Local dir or HF repo id (auto-downloads to test_models/sources/)",
      placeholder="test_models/sources/Qwen3.8-27B or Qwen/Qwen3.8-27B"),
    f("bits", "Bits per weight", "choice", default="3.5",
      choices=["2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"],
      help="Target average bitrate (-b)"),
    f("hq", "High quality (-hq)", "bool", default=True,
      help="Increase bitrate of attention / shared-expert layers"),
    f("name", "Job name", "str", help="Friendly name for the job record"),
    f("out", "Output dir (-o)", "path",
      help="Default: test_models/<model>-exl3-<bits>bpw"),
    f("work", "Work dir (-w)", "path", help="Default: <out>.work"),
    f("cal_rows", "Cal rows (-cr)", "int", help="Calibration rows (default 250; lower to save RAM)"),
    f("cal_cols", "Cal cols (-cc)", "int", help="Calibration cols (default 2048; lower to save RAM)"),
    f("cal_data", "Cal trace (-cd)", "path",
      help="Calibration trace JSON (e.g. test_models/sources/cal_trace_qwen38_27b.json)"),
    f("head_bits", "Head bits (-hb)", "int", help="Output layer bits, default 6"),
    f("shard_size", "Shard size MB (-ss)", "int", help="Max shard size, default 8192"),
    f("device", "Devices (-d)", "str", help="GPU list, default '0'"),
    f("checkpoint_interval", "Checkpoint interval s (-cpi)", "int", default=900,
      help="Seconds between resumable checkpoints (convert.py -cpi)"),
    f("draft_model_dir", "Draft model dir", "path",
      help="Optional: package this draft into <out>/draft + dspark.json after conversion"),
]

SPECS = [
    # ------------------------------------------------------------------ quant
    {
        "key": "quant",
        "group": "quantize",
        "label": "Quantize model → EXL3",
        "description": "tools/quantize_model.py — resolve source (local/HF), classify format "
                       "(bf16 / Qwen-fp8 / DeepSeek-native), preflight exllamav3 support, run "
                       "convert.py with resumable checkpoints, record in jobs.jsonl.",
        "fields": QUANT_FIELDS,
    },
    {
        "key": "quant_preflight",
        "group": "quantize",
        "label": "Preflight only (download + support check)",
        "description": "quantize_model.py --dry-run: download if needed, classify the "
                       "checkpoint format, verify exllamav3 supports the architecture.",
        "fields": [
            f("model", "Source model", "path", required=True,
              placeholder="Qwen/Qwen3.8-27B or local dir"),
        ],
    },
    {
        "key": "quant_resume",
        "group": "quantize",
        "label": "Resume interrupted conversion",
        "description": "quantize_model.py --resume — restore all settings from the work dir "
                       "checkpoint (convert.py -w WORK -r).",
        "fields": [
            f("work", "Work dir (-w)", "path", required=True,
              placeholder="test_models/<out>.work"),
            f("out", "Output dir (-o)", "path",
              help="Only needed together with a draft attach"),
            f("draft_model_dir", "Draft model dir", "path",
              help="Optional: attach draft after the resumed conversion finishes"),
        ],
    },
    {
        "key": "attach_draft",
        "group": "quantize",
        "label": "Attach draft to existing EXL3 dir",
        "description": "quantize_model.py --attach-draft — copy a DSpark/DFlash2 draft into "
                       "<out>/draft/ and write a dspark.json serving manifest.",
        "fields": [
            f("out", "EXL3 model dir (-o)", "path", required=True),
            f("draft_model_dir", "Draft model dir", "path", required=True),
        ],
    },
    {
        "key": "convert_raw",
        "group": "quantize",
        "label": "Advanced: raw convert.py",
        "script": "convert.py",
        "description": "Direct convert.py with the full flag surface (-i/-w/-o required for a "
                       "new job, or -w + -r to resume). Use the friendlier 'Quantize model' "
                       "entry above unless you need flags not exposed there.",
        "fields": [
            f("in_dir", "Input dir (-i)", "path", help="Unquantized HF model dir"),
            f("work_dir", "Work dir (-w)", "path"),
            f("out_dir", "Output dir (-o)", "path"),
            f("bits", "Bits (-b)", "float"),
            f("hq", "-hq", "bool", help="Protect attention/shared-expert layers"),
            f("resume", "Resume (-r)", "bool", help="Resume interrupted job from work dir"),
            f("head_bits", "Head bits (-hb)", "int", help="default 6"),
            f("mtp_bits", "MTP bits (-mb)", "int", help="default 4"),
            f("vision_bits", "Vision bits (-vb)", "int", help="1-8, or 16 unquantized"),
            f("cal_rows", "Cal rows (-cr)", "int", help="default 250"),
            f("cal_cols", "Cal cols (-cc)", "int", help="default 2048"),
            f("cal_data", "Cal data (-cd)", "path", help="Calibration trace JSON"),
            f("checkpoint_interval", "Checkpoint s (-cpi)", "int", help="default 120"),
            f("shard_size", "Shard MB (-ss)", "int", help="default 8192"),
            f("devices", "Devices (-d)", "str", default="0", help="e.g. 0,1,2"),
            f("device_ratios", "Device ratios (-dr)", "str", help="e.g. 2,2,4"),
            f("codebook", "Codebook (-cb)", "choice", default="mul1",
              choices=["mul1", "mcg", "3inst"]),
            f("out_scales", "Out scales (--out_scales)", "choice", default="always",
              choices=["always", "never", "auto"]),
            f("verbose", "Verbose (-v)", "bool"),
            f("max_module", "Max modules (--max_module)", "int", help="debug: stop after N modules"),
            f("override_anyway", "--override_anyway", "bool",
              help="Allow resuming even when overriding breaking settings"),
        ],
    },
    # ------------------------------------------------------------------ serve
    {
        "key": "serve",
        "group": "serve",
        "label": "Serve model (OpenAI-compatible)",
        "description": "tools/serve_openai.py — /v1/chat/completions + /v1/models + /health, "
                       "streaming, tool calling, speculative draft. Requires the GPU to be "
                       "free of other big workloads (unified memory).",
        "fields": [
            f("model", "Model dir (-m)", "path",
              default="test_models/Qwen3.8-27B-exl3-3.5bpw-wm", required=True),
            f("draft_model", "Draft dir (-dm)", "path",
              default="test_models/Qwen3.8-27B-exl3-3.5bpw-wm/draft-dflash2",
              help="'none' disables the draft"),
            f("port", "Port (-p)", "int", default=8888),
            f("host", "Host (--host)", "str", default="0.0.0.0",
              help="127.0.0.1 for local-only"),
            f("grid_size", "Grid size (-gs)", "int", default=110),
            f("cache_size", "KV cache tokens (-cs)", "int", default=65536),
            f("cache_quant", "KV cache quant (-cq)", "str",
              help="e.g. '8' or '8,4' (k_bits,v_bits)"),
            f("cpu_cache_size", "CPU cache GB (-ccs)", "float",
              help="Second-tier CPU cache, pages spill when GPU cache is full"),
            f("dspark_conf", "EXL3_DSPARK_CONF", "env",
              help="Optional env: DSpark draft confidence threshold (ignored by DFlash2)"),
        ],
    },
]

# ---------------------------------------------------------------- eval scripts

EVAL_SPECS = [
    {
        "key": "eval_ppl",
        "group": "eval",
        "label": "Perplexity (WikiText-2)",
        "script": "eval/ppl.py",
        "description": "eval/ppl.py — WikiText-2 perplexity; EXL3, HF Transformers (-hf), or "
                       "GGUF-equivalent mode (-g).",
        "fields": mi_fields(cache_default=None) + [
            f("rows", "Rows (-r)", "int", default=100),
            f("length", "Tokens per row (-l)", "int", default=2048),
            f("gguf", "GGUF-equivalent mode (-g)", "bool", help="ignores -r/-l"),
            f("ctx_size", "GGUF ctx size (-c)", "int", default=512),
            f("hf", "Transformers backend (-hf)", "bool"),
            f("hf_device", "HF device (-hf_d)", "int"),
            f("hf_tight", "HF force FP16 (-hf_t)", "bool"),
            f("hf_fp32", "HF force FP32 (-hf_fp32)", "bool"),
            f("gen_prompt", "Prepend chat prompt (-gp)", "bool"),
        ],
    },
    {
        "key": "eval_mmlu",
        "group": "eval",
        "label": "MMLU / MMLU-Redux",
        "script": "eval/mmlu.py",
        "description": "eval/mmlu.py — MMLU benchmark incl. MMLU-Redux variants.",
        "fields": mi_fields(cache_default=8192) + [
            f("fewshot", "Fewshot examples (-fs)", "int", default=5, help="max 5"),
            f("subjects", "Subjects (-sub)", "str", default="all",
              help="comma-separated subjects or 'all'"),
            f("shuffle", "Shuffle choices (-shf)", "bool"),
            f("skip_subjects", "Skip N subjects (-skip)", "int"),
            f("max_q", "Max questions/subject (-mqps)", "int"),
            f("random_sample", "Random sample N (-r)", "int"),
            f("redux", "MMLU-Redux (-redux)", "choice", choices=["", "1", "2"],
              help="Redux v1 or v2.0"),
        ],
    },
    {
        "key": "eval_humaneval",
        "group": "eval",
        "label": "HumanEval (pass@k)",
        "script": "eval/humaneval.py",
        "description": "eval/humaneval.py — HumanEval pass@k sampling to .jsonl.",
        "fields": mi_fields(cache_default=8192) + [
            f("output", "Output .jsonl (-o)", "str", required=True,
              placeholder="runs/2026/…/humaneval.jsonl"),
            f("samples", "Samples per task (-spt)", "int", default=200),
            f("prompt_format", "Prompt format (-pf)", "choice",
              choices=["", "raw", "qwen3", "chatml", "deepseek", "llama3", "llama",
                       "mistral", "gemma", "granite", "reka", "minimax"],
              help="instruct format"),
            f("eval_after", "Run evaluator after (-e)", "bool"),
            f("temperature", "Temperature (-temp)", "float", default=0.6),
            f("min_p", "Min-P (-minp)", "float", default=0.0),
            f("top_k", "Top-K (-topk)", "int", default=0),
            f("top_p", "Top-P (-topp)", "float", default=0.6),
            f("max_tokens", "Max tokens (--max_tokens)", "int", default=1024),
            f("max_batch", "Max batch (-mbs)", "int", default=256),
            f("verbose", "Verbose (-v)", "bool"),
        ],
    },
    {
        "key": "eval_ifbench",
        "group": "eval",
        "label": "IFBench",
        "script": "eval/ifbench.py",
        "description": "eval/ifbench.py — instruction-following benchmark.",
        "fields": mi_fields(cache_default=65536) + [
            f("output", "Output .jsonl (-o)", "str", required=True),
            f("eval_after", "Eval after (-e)", "bool"),
            f("max_tokens", "Max tokens (-mt)", "int", default=16384),
            f("max_batch", "Max batch (-mbs)", "int", default=64),
            f("think", "Thinking on (-think)", "bool"),
            f("nothink", "Thinking off (-nothink)", "bool"),
            f("limit", "Limit questions (-limit)", "int", help="0 = all"),
        ],
    },
    {
        "key": "eval_bbeh",
        "group": "eval",
        "label": "BBeh mini (BigBench extra-hard)",
        "script": "eval/bbeh_mini.py",
        "description": "eval/bbeh_mini.py — BigBench Extra Hard, mini sample set.",
        "fields": mi_fields(cache_default=65536) + [
            f("output", "Output .jsonl (-o)", "str"),
            f("max_tokens", "Max tokens (-mt)", "int", default=16384),
            f("max_batch", "Max batch (-mbs)", "int", default=64),
            f("think", "Thinking on (-think)", "bool"),
            f("nothink", "Thinking off (-nothink)", "bool"),
            f("limit", "Limit questions (-limit)", "int", help="0 = all"),
        ],
    },
    {
        "key": "eval_longctx",
        "group": "eval",
        "label": "Long-context comprehension",
        "script": "eval/longctx.py",
        "description": "eval/longctx.py — summary / translation / QA / corruption / name "
                       "extraction at long context (supports draft model).",
        "fields": mi_fields(cache_default=65536, draft=True) + [
            f("extra_long", "Extra long (-xl)", "bool", help="harder tests, needs ~200k cache"),
        ],
    },
    {
        "key": "eval_perf",
        "group": "eval",
        "label": "Perf (prefill + decode throughput)",
        "script": "eval/perf.py",
        "description": "eval/perf.py — prefill + generation throughput benchmark.",
        "fields": mi_fields(cache_default=32768) + [
            f("max_length", "Max length (-max_length)", "int", default=32768),
            f("chunk_size", "Chunk size (-chunk_size)", "int", default=4096),
            f("skip_prefill", "Skip prefill (-spf)", "bool"),
            f("skip_gen", "Skip generation (-sg)", "bool"),
            f("skip_warmup", "Skip warmup (-swu)", "bool"),
            f("short_prefill", "Short-prefill mode (-short)", "bool"),
        ],
    },
    {
        "key": "eval_specdecode",
        "group": "eval",
        "label": "Spec-decode benchmark",
        "script": "eval/spec_decode.py",
        "description": "eval/spec_decode.py — baseline vs n-gram vs draft-model speculative "
                       "decoding.",
        "fields": mi_fields(cache_default=49152, draft=True) + [
            f("no_baseline", "Skip baseline (-nbl)", "bool"),
            f("ngram_min", "N-gram min match (-ngram_min)", "int"),
            f("ngram_len", "N-gram draft length (-ngram_len)", "int", default=4),
            f("max_new_tokens", "Tokens per round (-tokens)", "int", default=1024),
            f("temperature", "Also sample w/ temp (-temp)", "bool"),
            f("single", "Single workload (-single)", "str", help="supports prefix*"),
            f("draft_stats", "Draft stats JSON (-dstats)", "str"),
            f("print_stats", "Print stats (-print)", "bool"),
        ],
    },
    {
        "key": "eval_diversity",
        "group": "eval",
        "label": "Sampling diversity",
        "script": "eval/diversity.py",
        "description": "eval/diversity.py — sampling diversity test.",
        "fields": mi_fields(cache_default=32768) + [
            f("samples", "Samples (-samples)", "int", default=50),
            f("max_tokens", "Max tokens (-max_tokens)", "int", default=2048),
            f("max_batch", "Max batch (-mbs)", "int", default=16),
        ],
    },
    {
        "key": "eval_modeldiff",
        "group": "eval",
        "label": "Model diff (quant error, KLD/top-K/PPL)",
        "script": "eval/model_diff.py",
        "description": "eval/model_diff.py — per-module quantization-error diff between two "
                       "models (streams one module at a time).",
        "fields": [
            f("model_a", "Model A (-ma)", "path", required=True, help="usually the quantized model"),
            f("model_b", "Model B (-mb)", "path", required=True, help="reference model"),
            f("rows", "Rows (-r)", "int", default=100),
            f("length", "Tokens/row (-l)", "int", default=2048),
            f("topk_max", "Top-K max (-tkm)", "int", default=5),
            f("device", "Device (-d)", "int", default=0),
            f("batch_size", "Batch (-bsz)", "int", default=1),
            f("gen_prompt", "Prepend chat prompt (-gp)", "bool"),
            f("no_reconstruct", "No GEMM reconstruct (-nr)", "bool"),
            f("cache_quant", "Sim cache quant A (-cq)", "str"),
            f("sweep", "Cache quant sweep (-cqs)", "bool"),
        ],
    },
    {
        "key": "eval_modeldiff_attrib",
        "group": "eval",
        "label": "Model diff attribution (module swap)",
        "script": "eval/model_diff_attrib.py",
        "description": "eval/model_diff_attrib.py — quantization error attribution by "
                       "single-module swap (loads both models fully).",
        "fields": [
            f("model_a", "Quantized model (-ma)", "path", required=True),
            f("model_b", "Reference model (-mb)", "path", required=True),
            f("rows", "Rows (-r)", "int", default=20),
            f("length", "Tokens/row (-l)", "int", default=2048),
            f("device", "Device (-d)", "int", default=0),
            f("level", "Granularity (-lv)", "choice", default="linear",
              choices=["block", "linear"]),
            f("iso", "Isotropic control (-iso)", "bool", help="2x runtime"),
            f("top", "Top contributors (-t)", "int", default=15),
            f("out", "Results JSON (-o)", "str"),
        ],
    },
    {
        "key": "eval_qbench",
        "group": "eval",
        "label": "Qbench (YAML project harness)",
        "script": "eval/qbench.py",
        "description": "eval/qbench.py — YAML-project-driven quantization comparison "
                       "(exllamav3 / transformers / llamacpp engines).",
        "fields": [
            f("project", "Project YAML", "path", required=True, positional=True,
              placeholder="eval/qbench_example.yaml"),
            f("device", "Device (-d)", "int", default=0),
        ],
    },
    {
        "key": "eval_prequant",
        "group": "eval",
        "label": "Pre-quant test (weight/state stats)",
        "script": "eval/prequant_test.py",
        "description": "eval/prequant_test.py — per-module pre-quantization stats, histograms "
                       "and PPL (module-streaming).",
        "fields": [
            f("model_dir", "Model dir (-m)", "path", required=True),
            f("device", "Device (-d)", "int", default=0),
            f("rows", "Rows (-r)", "int", default=10),
            f("bins", "Histogram bins (-hb)", "int", default=160),
            f("no_histogram", "No histogram (-nh)", "bool"),
            f("bos", "Add BOS (-bos)", "bool"),
            f("skip_tokens", "Skip tokens (-skip)", "int"),
            f("from_layer", "From layer (-fl)", "int"),
            f("to_layer", "To layer (-tl)", "int"),
            f("no_inspect", "No module inspect (-nim)", "bool"),
            f("gen_prompt", "Prepend chat prompt (-gp)", "bool"),
        ],
    },
]

# --------------------------------------------------------------- tool scripts

TOOL_SPECS = [
    {
        "key": "dequant_fp8",
        "group": "tools",
        "label": "Dequantize DeepSeek fp8/int8 → bf16",
        "script": "tools/fp8_to_bf16.py",
        "description": "tools/fp8_to_bf16.py — convert a DeepSeek-native fp8/int8 checkpoint "
                       "to bf16 so convert.py can consume it. CPU/disk only.",
        "fields": [
            f("in_dir", "Input dir (-i)", "path", required=True),
            f("out_dir", "Output dir (-o)", "path", required=True),
            f("shard_gb", "Shard GB (--shard_gb)", "float", default=4.0),
            f("keep_fp8", "Keep fp8 attn tensors (--keep-fp8)", "bool",
              help="Leave carried F8_E4M3 attention tensors as-is (smaller output)"),
        ],
    },
    {
        "key": "dequant_nvfp4",
        "group": "tools",
        "label": "Dequantize NVFP4 → bf16",
        "script": "tools/nvfp4_to_bf16.py",
        "description": "tools/nvfp4_to_bf16.py — NVFP4/ModelOpt to bf16 (weight-format loss "
                       "measurement). CPU/disk only.",
        "fields": [
            f("in_dir", "Input dir (-i)", "path", required=True),
            f("out_dir", "Output dir (-o)", "path", required=True),
            f("source", "bf16 source (--source)", "path",
              default="test_models/sources/Qwen3.8-27B",
              help="bf16 source for convention probing"),
            f("probe_only", "Probe only (--probe-only)", "bool",
              help="only run scale/nibble convention probe"),
        ],
    },
    {
        "key": "gen_cal_trace",
        "group": "tools",
        "label": "Generate calibration trace",
        "script": "tools/gen_cal_trace.py",
        "description": "tools/gen_cal_trace.py — workload-matched calibration trace generator "
                       "(resumable). Env: M2_TARGET_DIR / M2_DRAFT_DIR.",
        "fields": [
            f("out", "Output JSON (--out)", "str",
              default="test_models/sources/cal_trace_qwen38_27b_workload.json"),
            f("target_tokens", "Target tokens (--target-tokens)", "int", default=620000),
            f("max_new_tokens", "Max new tokens (--max-new-tokens)", "int", default=3072),
            f("reasoning_effort", "Reasoning effort", "choice",
              choices=["", "xhigh", "medium", "low"]),
            f("no_draft", "No draft (--no-draft)", "bool"),
            f("num_draft_tokens", "Draft tokens (--num-draft-tokens)", "int", default=7),
            f("smoke", "Smoke test (--smoke)", "bool", help="3 prompts x 96 tokens to /tmp"),
            f("finalize", "Finalize only (--finalize)", "bool",
              help="assemble JSON from existing jsonl rows"),
            f("cal_rows", "Cal rows (--cal-rows)", "int", default=250),
            f("cal_cols", "Cal cols (--cal-cols)", "int", default=2048),
            f("M2_TARGET_DIR", "M2_TARGET_DIR", "env", help="target model dir"),
            f("M2_DRAFT_DIR", "M2_DRAFT_DIR", "env", help="draft model dir"),
        ],
    },
    {
        "key": "dspark_m4_bench",
        "group": "tools",
        "label": "DSpark M4 bench (acceptance + speed)",
        "script": "tools/dspark_m4_bench.py",
        "description": "tools/dspark_m4_bench.py — per-workload speculative acceptance + decode "
                       "speed. Env: M2_TARGET_DIR / M2_DRAFT_DIR / M4_EXTRA_ARGS.",
        "fields": [
            f("workloads", "Workloads (--workloads)", "str",
              default="humaneval,gsm8k,math500,mbpp,mt-bench,alpaca"),
            f("num_prompts", "Prompts (--num-prompts)", "int", default=24),
            f("max_new_tokens", "Max new tokens (--max-new-tokens)", "int", default=512),
            f("baseline", "Baseline (--baseline)", "bool", help="run no-draft baseline"),
            f("temperature", "Temperature (--temperature)", "float", default=0.6),
            f("top_k", "Top-K (--top-k)", "int", default=20),
            f("top_p", "Top-P (--top-p)", "float", default=0.95),
            f("tag", "Tag (--tag)", "str", help="output file tag"),
            f("sweep_conf", "Sweep conf (--sweep-conf)", "bool",
              help="EXL3_DSPARK_CONF threshold sweep on first workload"),
            f("M2_TARGET_DIR", "M2_TARGET_DIR", "env", help="target model dir"),
            f("M2_DRAFT_DIR", "M2_DRAFT_DIR", "env", help="draft model dir"),
        ],
    },
    {
        "key": "decode_bench",
        "group": "tools",
        "label": "Decode bench (CPU affinity A/B)",
        "script": "tools/decode_bench.py",
        "description": "tools/decode_bench.py — steady-state decode tok/s A/B of CPU affinity "
                       "(model dirs hardcoded in script; affinity via EXL3_AFFINITY).",
        "fields": [
            f("runs", "Runs (--runs)", "int", default=3),
            f("profile", "cProfile probe (--profile)", "bool"),
            f("EXL3_AFFINITY", "EXL3_AFFINITY", "env", default="5-9,15-19",
              help="CPU affinity mask"),
        ],
    },
    {
        "key": "accept_probe",
        "group": "tools",
        "label": "Accept probe (greedy vs sampled)",
        "script": "tools/accept_probe.py",
        "description": "tools/accept_probe.py — acceptance-chain stats, greedy vs sampled "
                       "decode. Config via env: TM (target dir), DM (draft dir), DRAFT_T.",
        "fields": [
            f("TM", "TM (target dir)", "env", required=True),
            f("DM", "DM (draft dir)", "env"),
            f("DRAFT_T", "DRAFT_T (draft temperature)", "env", default="0.6",
              help="0 = greedy only"),
        ],
    },
    {
        "key": "longctx_passkey",
        "group": "tools",
        "label": "Long-ctx passkey (needle recall)",
        "script": "tools/longctx_passkey.py",
        "description": "tools/longctx_passkey.py — needle-in-haystack passkey recall at depth. "
                       "Env: TM (target dir). Positional args: ctx tokens, depths %, KV bits.",
        "fields": [
            f("ctx", "Context tokens", "int", default=300000, positional=True),
            f("depths", "Depths %% CSV", "str", default="10,50,90", positional=True),
            f("kv_bits", "KV cache bits (-cq)", "str", default="8", positional=True),
            f("TM", "TM (target dir)", "env", required=True),
        ],
    },
    {
        "key": "longctx_sanity",
        "group": "tools",
        "label": "Long-ctx sanity (wm-1m bundle)",
        "script": "tools/longctx_sanity.py",
        "description": "tools/longctx_sanity.py — wm-1m bundle + quant KV + DFlash2 sanity "
                       "check. Single optional positional: -cq bits.",
        "fields": [
            f("cq", "KV cache bits", "str", default="4", positional=True),
        ],
    },
    {
        "key": "kernel_profile",
        "group": "tools",
        "label": "Kernel profile (torch.profiler)",
        "script": "tools/kernel_profile.py",
        "description": "tools/kernel_profile.py — CUDA kernel table for one verify forward. "
                       "Config via env: PROFILE_AT, PROFILE_WHAT, DM.",
        "fields": [
            f("PROFILE_AT", "PROFILE_AT (step)", "env", default="30"),
            f("PROFILE_WHAT", "PROFILE_WHAT", "env", default="target",
              help="target | draft | verify_accept"),
            f("DM", "DM (draft dir)", "env"),
        ],
    },
    {
        "key": "step_breakdown",
        "group": "tools",
        "label": "Step breakdown (wall vs GPU time)",
        "script": "tools/step_breakdown.py",
        "description": "tools/step_breakdown.py — per-step decode breakdown. Env: DM "
                       "(draft dir).",
        "fields": [
            f("DM", "DM (draft dir)", "env"),
        ],
    },
    {
        "key": "cudagraph_probe",
        "group": "tools",
        "label": "CUDA-graph probe (parity + speed)",
        "script": "tools/cudagraph_probe.py",
        "description": "tools/cudagraph_probe.py — CUDA-graph target verify parity + speed "
                       "probe.",
        "fields": [
            f("mode", "Mode (--mode)", "choice", default="both",
              choices=["parity", "speed", "both"]),
            f("max_new", "Max new (--max-new)", "int", default=160),
        ],
    },
    {
        "key": "gemm_audit",
        "group": "tools",
        "label": "GEMM audit (decode-shape bandwidth)",
        "script": "tools/gemm_audit.py",
        "description": "tools/gemm_audit.py — per-module EXL3 GEMM bandwidth audit at decode "
                       "shapes (model dirs hardcoded in script).",
        "fields": [],
    },
]

SPECS += EVAL_SPECS + TOOL_SPECS
