# ExLlamaV3 fork (MiaAI Lab)

Fork of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) — EXL3
inference for local LLMs — adding what we needed to serve **Qwen3.8-27B**
on DGX Spark (GB10, aarch64) and 24 GB GPUs.

## What this fork adds

- **Speculative decoding**: DFlash2 and DSpark diffusion draft models, and
  MTP drafting for checkpoints with a multi-token-prediction head
  (Qwen3.8-27B ships one — `--mtp`).
- **Quantized KV cache**: NVFP4 (E2M1 + block-16 E4M3 scales, ~4.5 bits/elem)
  and FP8 lanes with online dequant in the Triton paged-attention kernels —
  measured lossless at generation level, ~3.5x more context per GB than fp16.
- **aarch64 / GB10 (sm_120 / sm_121) support**: builds and runs from source on
  DGX Spark (no prebuilt wheels exist upstream).
- **`tools/serve_openai.py`** — OpenAI-compatible server: chat completions
  (stream + non-stream), Qwen tool calling with typed arguments, batch-1
  speculative decode.
- **`tools/webconsole.py`** — web UI over the CLI: quantize, serve, chat
  playground, eval harnesses, job registry with live logs.
- **`start.sh` / `stop.sh`** — self-bootstrapping launcher: venv, GPU torch,
  engine build, HF weight download, YaRN long-context config swap.
- **`quantize.sh`** — GB10-friendly launcher for the EXL3 quantizer.

## Quick start (serving)

Easiest: the deployment kit —
[MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
(one command: downloads the weights, builds everything, serves the OpenAI
API — see its README for the full instructions).

From this repo:

```bash
./start.sh    # creates .env from .env.example on first run; edit and rerun
```

Library/API use follows upstream; the upstream README is preserved as
`UPSTREAM_README.md`.

## License

MIT (as upstream). Fork changes: MIT.
