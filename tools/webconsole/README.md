# EXL3 web console

A complete web UI over this repo's CLI surface. Everything you can do from
the command line can be done here, as thin wrappers over the same scripts —
no duplicated logic, same flags, same defaults.

```bash
.venv/bin/python tools/webconsole.py --host 0.0.0.0 --port 7861
# open http://<host>:7861
```

No new dependencies: it uses `aiohttp`, the same stack `tools/serve_openai.py`
already uses. The legacy Gradio launcher (`tools/webui.py`) is untouched.

## Coverage

| section    | wraps |
|------------|-------|
| Quantize   | `tools/quantize_model.py` (start / preflight / resume / attach-draft) + raw `convert.py` with the full flag surface |
| Serve      | `tools/serve_openai.py` (start / stop / health probe) |
| Chat       | playground for a running `serve_openai` endpoint (streaming SSE, reasoning, tool calls) via a console proxy |
| Evaluate   | all 13 `eval/*.py` harnesses with their full CLI options (ppl, mmlu, humaneval, ifbench, bbeh_mini, longctx, perf, spec_decode, diversity, model_diff, model_diff_attrib, qbench, prequant_test) |
| Tools      | `tools/*.py`: fp8/nvfp4 dequant, gen_cal_trace, dspark_m4_bench, decode_bench, accept_probe, longctx_passkey, longctx_sanity, kernel_profile, step_breakdown, cudagraph_probe, gemm_audit |
| Models     | browser for `test_models/` + `test_models/sources/` (kind/size/draft badges) |
| Jobs       | unified registry (`runs/webui/jobs.jsonl`), live logs, kill; legacy `jobs.jsonl` records included |
| System     | GPU / RAM / disk / versions |

## Design

- `tools/webconsole.py` — aiohttp app: REST API + static serving + chat proxy.
- `tools/webconsole/specs.py` — the CLI capability manifest. One entry per
  script; each field maps 1:1 to a CLI flag (flags are derived from the label
  text, e.g. `Cache quant (-cq)`). Env-driven scripts (`TM`, `DM`,
  `EXL3_AFFINITY`, …) use `env` fields; positional-arg scripts use
  `positional` fields. The frontend renders forms from this manifest and the
  server translates params → argv, so the UI can never drift from the CLI.
- `tools/webconsole/jobman.py` — job runner: every action spawns a detached
  subprocess (`start_new_session`), so jobs survive a console restart and are
  killable by process group. Append-only JSONL registry; logs under
  `runs/webui/logs/`.
- `tools/webconsole/static/` — dependency-free SPA (vanilla JS + CSS, dark
  theme, no build step, no CDN).

## Notes

- The console itself is light; it never touches the GPU. Heavy work always
  runs as a child of the repo's own scripts.
- Quant jobs run `quantize_model.py` with `--log` routed next to the job log;
  the log viewer tails both files combined.
- `TORCH_CUDA_ARCH_LIST` / `CUDA_HOME` are pinned for this GB10 box on quant
  jobs, mirroring `quantize.sh`.
- Bind `--host 127.0.0.1` (default) for local-only, `0.0.0.0` to reach the
  console from the LAN. There is no auth — treat it like the other dev
  servers on this box.
